#include <sstream>
#include <string>
#include <vector>

#include "ramulator/controller/addr_mapper/i_addr_mapper.h"
#include "ramulator/controller/addr_mapper/addr_mapper_base.h"

namespace Ramulator {

// PFM region mapper: a different address layout for each memory region.
//
// The physical address space is split into regions of 2^region_bits bytes
// (default 21 = 2 MB superpages). Each region uses one entry of `layouts`,
// chosen by `region_layout[region index]`; regions past the end of the table
// use `default_layout`. A value of -1 means the stock RoBaRaCoCh order.
//
// A layout lists which DRAM level each line-index bit of the region selects,
// from the least significant bit upward, as "level count" tokens, e.g.
//   "bg1 col6 bank1 col1 bank1 bg1 row4"
// Levels: rank, bg (bank group), bank, row, col. Bank-group, bank, rank and
// column must use exactly their device bit count; the region index is placed
// above the layout's row bits.
class PFMRegionMapper : public IAddrMapper, public AddrMapperBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IAddrMapper, PFMRegionMapper, AddrMapperBase, "PFMRegionMapper")

  struct BitDest {
    int level;  // index into addr_vec (1 = first level after channel)
    int shift;  // bit position inside that level's value
  };
  struct Layout {
    std::vector<BitDest> bits;  // one entry per line-index bit inside the region
    int row_bits = 0;
  };

  int m_region_bits = 21;
  int m_default_layout = -1;
  std::vector<std::string> m_layout_strings;
  std::vector<int> m_region_layout;

  std::vector<Layout> m_layouts;
  int m_local_bits = 0;  // line-index bits per region

  size_t s_region_mapped = 0;
  size_t s_default_mapped = 0;

  void init() override;
  void apply(Request& req) override;

  int level_index(const std::string& token);
  Layout parse_layout(const std::string& s, int idx);
  void apply_default(Request& req);
};

int PFMRegionMapper::level_index(const std::string& token) {
  const auto& spec = *m_ctrl->m_device.m_spec;
  std::string name;
  if (token == "rank") name = "Rank";
  else if (token == "bg") name = "BankGroup";
  else if (token == "bank") name = "Bank";
  else if (token == "row") name = "Row";
  else if (token == "col") name = "Column";
  else throw std::runtime_error(get_name() + ": unknown level '" + token + "' (use rank, bg, bank, row, col)");
  return spec.get_level_id(name);  // channel is level 0, so this is already the addr_vec index
}

PFMRegionMapper::Layout PFMRegionMapper::parse_layout(const std::string& s, int idx) {
  Layout lay;
  std::vector<int> used(m_num_mapped_levels + 1, 0);
  std::istringstream in(s);
  std::string tok;
  while (in >> tok) {
    size_t p = tok.find_first_of("0123456789");
    if (p == std::string::npos || p == 0) {
      throw std::runtime_error(get_name() + ": bad token '" + tok + "' in layout " + std::to_string(idx));
    }
    int level = level_index(tok.substr(0, p));
    int n = std::stoi(tok.substr(p));
    for (int k = 0; k < n; k++) {
      lay.bits.push_back({level, used[level]++});
    }
  }
  if ((int)lay.bits.size() != m_local_bits) {
    throw std::runtime_error(get_name() + ": layout " + std::to_string(idx) + " has " +
                             std::to_string(lay.bits.size()) + " bits, the region needs " +
                             std::to_string(m_local_bits));
  }
  for (int i = 0; i < m_num_mapped_levels; i++) {
    int level = i + 1;
    if (i == m_row_idx) {
      if (used[level] > m_addr_bits[i]) {
        throw std::runtime_error(get_name() + ": layout " + std::to_string(idx) + " uses too many row bits");
      }
      lay.row_bits = used[level];
    } else if (used[level] != m_addr_bits[i]) {
      throw std::runtime_error(get_name() + ": layout " + std::to_string(idx) + " gives level " +
                               std::to_string(level) + " " + std::to_string(used[level]) + " bits, device has " +
                               std::to_string(m_addr_bits[i]));
    }
  }
  return lay;
}

void PFMRegionMapper::init() {
  AddrMapperBase::init();
  RAMULATOR_PARSE_PARAM(m_region_bits, int, "region_bits").default_val(21);
  RAMULATOR_PARSE_PARAM(m_layout_strings, std::vector<std::string>, "layouts").required();
  RAMULATOR_PARSE_PARAM(m_region_layout, std::vector<int>, "region_layout").default_val(std::vector<int>{});
  RAMULATOR_PARSE_PARAM(m_default_layout, int, "default_layout").default_val(-1);

  m_local_bits = m_region_bits - (int)m_tx_offset;
  if (m_local_bits <= 0) {
    throw std::runtime_error(get_name() + ": region_bits must be larger than the transaction offset");
  }
  for (size_t i = 0; i < m_layout_strings.size(); i++) {
    m_layouts.push_back(parse_layout(m_layout_strings[i], (int)i));
  }
  auto check = [&](int id) {
    if (id < -1 || id >= (int)m_layouts.size()) {
      throw std::runtime_error(get_name() + ": layout id " + std::to_string(id) + " out of range");
    }
  };
  for (int id : m_region_layout) check(id);
  check(m_default_layout);

  m_stats.add("region_mapped", s_region_mapped);
  m_stats.add("default_mapped", s_default_mapped);
}

void PFMRegionMapper::apply_default(Request& req) {
  // Same as RoBaRaCoCh
  Addr_t addr = req.intra_channel_addr >> m_tx_offset;
  req.addr_vec[m_col_idx + 1] = slice_lower_bits(addr, m_addr_bits[m_col_idx]);
  for (int i = 0; i <= m_row_idx; i++) {
    req.addr_vec[i + 1] = slice_lower_bits(addr, m_addr_bits[i]);
  }
}

void PFMRegionMapper::apply(Request& req) {
  req.addr_vec.resize(m_num_mapped_levels + 1, -1);
  Addr_t line = req.intra_channel_addr >> m_tx_offset;
  Addr_t region = line >> m_local_bits;
  int id = region < (Addr_t)m_region_layout.size() ? m_region_layout[region] : m_default_layout;
  if (id < 0) {
    apply_default(req);
    s_default_mapped++;
    return;
  }
  const Layout& lay = m_layouts[id];
  for (int i = 1; i <= m_num_mapped_levels; i++) {
    req.addr_vec[i] = 0;
  }
  for (int j = 0; j < m_local_bits; j++) {
    if ((line >> j) & 1) {
      req.addr_vec[lay.bits[j].level] |= 1 << lay.bits[j].shift;
    }
  }
  req.addr_vec[m_row_idx + 1] |= static_cast<int>(region << lay.row_bits);
  s_region_mapped++;
}

}  // namespace Ramulator
