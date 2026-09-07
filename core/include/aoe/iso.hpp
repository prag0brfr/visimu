// Author: Paulo R. A. Gilz
//
// ISO parameter grouping.
//
// An "ISO" is a parameter held constant within a simulation (Mach number, mass,
// altitude...). Runs are filtered and coloured by these values, and when a sweep
// produces too many distinct values to be legible they are bucketed into
// intervals.
//
// Pure data, with no dialogue in it. Interval arithmetic that shares a file
// with the popup that presents it cannot be tested or reused outside a live UI,
// so the grouping logic lives here and the dialogue lives in Python.

#pragma once

#include <string>
#include <vector>

namespace aoe {

struct IsoGroup {
    std::string name;
    bool active = true;
    float min_bound = 0.0f;
    float max_bound = 0.0f;
};

// Evenly spaced intervals spanning the observed value range. `values` need not
// be sorted. Returns an empty vector when there is nothing to group.
std::vector<IsoGroup> iso_auto_group(const std::vector<float>& values, int group_count);

// True when `value` falls inside any active group. Bounds are inclusive at both
// ends; the last group's upper bound is treated as closed so the maximum sample
// is never dropped by floating-point edge effects.
bool iso_value_active(float value, const std::vector<IsoGroup>& groups);

// Name of the group containing `value`, or an empty string when none matches.
std::string iso_group_name(float value, const std::vector<IsoGroup>& groups);

// Index of the group containing `value`, or -1. Used to pick a colour per group.
int iso_group_index(float value, const std::vector<IsoGroup>& groups);

// Recomputes bounds after new files arrive. mode 0 = auto (regenerate evenly
// spaced intervals over the new range), mode 1 = manual (keep the user's bounds
// and only refresh the generated names).
void iso_reevaluate(int mode, int group_count, std::vector<IsoGroup>& groups,
                    const std::vector<float>& values);

// How an ISO value is written wherever one is shown: a row of the ISO tree, a
// legend entry, an interval's generated name.
//
// Two decimals, trailing zeros trimmed -- 0.750000 reads as 0.75 and 3.000000
// as 3 -- and scientific notation at both ends of the range where two decimals
// stop being honest.
//
// The small end is decided by rounding rather than by a threshold, because the
// threshold is the subtle part: a value arrives as a float32, so the literal
// 0.01 is stored as 0.00999999978, and comparing *that* against 1e-2 sends
// 0.01 itself to "1e-2" -- the one value a Mach or tolerance sweep is most
// likely to contain. Asking instead whether the rounded form is still non-zero
// puts the boundary where it belongs: fixed for as long as two decimals say
// something, scientific once they would say "0".
//
// The large end is a threshold, and it is about the label rather than the
// number: past a million the integer part alone is longer than the whole value
// written as a power of ten. These appear on a tree row and inside a plot
// legend, where every character is one the data does not get.
constexpr int ISO_DECIMALS = 2;
constexpr double ISO_ROUND_SCALE = 100.0;   // 10^ISO_DECIMALS
constexpr double ISO_FIXED_MAX = 1e6;

std::string iso_format_value(float value);

}  // namespace aoe
