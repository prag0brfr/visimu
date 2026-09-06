// Author: Paulo R. A. Gilz
#include "aoe/iso.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace aoe {
namespace {

bool finite_min_max(const std::vector<float>& values, float* lo, float* hi) {
    bool seen = false;
    for (float v : values) {
        if (std::isnan(v) || std::isinf(v)) continue;
        if (!seen) {
            *lo = *hi = v;
            seen = true;
        } else if (v < *lo) {
            *lo = v;
        } else if (v > *hi) {
            *hi = v;
        }
    }
    return seen;
}

std::string interval_name(float lo, float hi) {
    return iso_format_value(lo) + " - " + iso_format_value(hi);
}

}  // namespace

std::string iso_format_value(float value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
    std::string text(buffer);

    if (text.find('.') != std::string::npos) {
        std::size_t last = text.find_last_not_of('0');
        if (last != std::string::npos && text[last] == '.') --last;
        text.erase(last + 1);
    }
    if (text == "-0") text = "0";
    return text;
}

std::vector<IsoGroup> iso_auto_group(const std::vector<float>& values, int group_count) {
    std::vector<IsoGroup> groups;
    if (values.empty() || group_count <= 0) return groups;

    float lo = 0.0f, hi = 0.0f;
    if (!finite_min_max(values, &lo, &hi)) return groups;

    // A sweep with a single distinct value yields one group rather than N
    // degenerate empty ones.
    if (!(hi > lo)) {
        IsoGroup single;
        single.min_bound = lo;
        single.max_bound = hi;
        single.name = iso_format_value(lo);
        groups.push_back(single);
        return groups;
    }

    const double span = static_cast<double>(hi) - static_cast<double>(lo);
    const double step = span / group_count;

    groups.reserve(static_cast<std::size_t>(group_count));
    for (int i = 0; i < group_count; ++i) {
        IsoGroup group;
        group.min_bound = static_cast<float>(lo + step * i);
        group.max_bound = (i + 1 == group_count)
                              ? hi
                              : static_cast<float>(lo + step * (i + 1));
        group.name = interval_name(group.min_bound, group.max_bound);
        group.active = true;
        groups.push_back(group);
    }
    return groups;
}

int iso_group_index(float value, const std::vector<IsoGroup>& groups) {
    if (std::isnan(value)) return -1;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        const IsoGroup& g = groups[i];
        const bool last = (i + 1 == groups.size());
        if (value >= g.min_bound && (last ? value <= g.max_bound : value < g.max_bound))
            return static_cast<int>(i);
    }
    // Guard against a value sitting a hair outside the last bound through
    // float rounding; without this the maximum sample of a sweep can vanish.
    if (!groups.empty()) {
        if (value <= groups.front().min_bound) return 0;
        if (value >= groups.back().max_bound) return static_cast<int>(groups.size()) - 1;
    }
    return -1;
}

bool iso_value_active(float value, const std::vector<IsoGroup>& groups) {
    if (groups.empty()) return true;  // no grouping configured: nothing filtered
    const int index = iso_group_index(value, groups);
    return index >= 0 && groups[static_cast<std::size_t>(index)].active;
}

std::string iso_group_name(float value, const std::vector<IsoGroup>& groups) {
    const int index = iso_group_index(value, groups);
    return index < 0 ? std::string() : groups[static_cast<std::size_t>(index)].name;
}

void iso_reevaluate(int mode, int group_count, std::vector<IsoGroup>& groups,
                    const std::vector<float>& values) {
    if (mode == 0) {
        // Auto: rebuild over the new range, carrying the active flags across by
        // position so a user's deselections survive a file being added.
        std::vector<bool> was_active;
        was_active.reserve(groups.size());
        for (const auto& g : groups) was_active.push_back(g.active);

        groups = iso_auto_group(values, group_count);
        for (std::size_t i = 0; i < groups.size() && i < was_active.size(); ++i)
            groups[i].active = was_active[i];
        return;
    }

    // Manual: the user owns the bounds. Only refresh names that still look
    // auto-generated, so hand-typed labels are preserved.
    for (auto& g : groups) {
        if (g.name.empty() || g.name.find(" - ") != std::string::npos)
            g.name = interval_name(g.min_bound, g.max_bound);
    }
}

}  // namespace aoe
