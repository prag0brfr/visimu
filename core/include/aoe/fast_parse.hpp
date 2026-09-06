// Author: Paulo R. A. Gilz
//
// Whitespace/number scanning primitives for the AOE text format.
//
// The hot loop of this application is "turn ~10^8 ASCII floats into a float
// array", so this is where the time actually goes. strtof() is correctly rounded
// but locale-aware and roughly an order of magnitude slower than what follows;
// std::from_chars is better but is not available for floats in every toolchain
// we need to support (notably libstdc++ < 11 and older Clang/libc++).
//
// Details that a hand-rolled float parser usually gets wrong, and that this one
// is careful about:
//   * Mantissa digits accumulate into a uint64 and are scaled once at the end,
//     rather than adding `digit * 0.1^k` per fractional digit. Accumulating per
//     digit loses precision progressively and produces visibly wrong low-order
//     bits.
//   * A bare '+' or '-' after the mantissa is not the start of an exponent,
//     except in the explicit Fortran form described below. Treating any of
//     e/E/+/- as an exponent marker misreads ordinary sign-separated columns.
//   * Fortran 'D'/'d' exponents (1.234D+05) are accepted.
//   * Fortran's exponent-without-E form (1.234+05, emitted when the exponent
//     field overflows the format width) is accepted.
//   * NaN / Inf / Fortran's '*****' overflow marker parse to NaN rather than
//     silently becoming 0, so bad samples are visible instead of plotting as
//     a spurious zero line.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace aoe {

inline constexpr bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}

inline constexpr bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

// Powers of ten that are exactly representable as doubles.
inline const double* pow10_table() noexcept {
    static const double table[] = {
        1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10,
        1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21,
        1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 1e30, 1e31, 1e32,
        1e33, 1e34, 1e35, 1e36, 1e37, 1e38};
    return table;
}

inline double scale_pow10(double value, int exponent) noexcept {
    const double* table = pow10_table();
    if (exponent >= 0) {
        while (exponent > 38) {
            value *= 1e38;
            exponent -= 38;
            if (std::isinf(value)) return value;
        }
        return value * table[exponent];
    }
    int neg = -exponent;
    while (neg > 38) {
        value /= 1e38;
        neg -= 38;
        if (value == 0.0) return value;
    }
    return value / table[neg];
}

// Parses one number starting at `p` (which may point at leading whitespace) and
// stores the address just past the consumed token in `*next`. Returns NaN for
// tokens that are not numbers, always advancing `*next` past the token so the
// caller cannot loop forever on malformed input.
inline float parse_float(const char* p, const char* end, const char** next) noexcept {
    while (p < end && is_space(*p)) ++p;
    if (p >= end) {
        *next = p;
        return 0.0f;
    }

    const char* token_start = p;
    bool negative = false;
    if (*p == '-') {
        negative = true;
        ++p;
    } else if (*p == '+') {
        ++p;
    }

    // Non-numeric tokens: NaN, Inf, and Fortran's '****' field-overflow marker.
    if (p < end && !is_digit(*p) && *p != '.') {
        while (p < end && !is_space(*p)) ++p;
        *next = p;
        (void)token_start;
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Accumulate all mantissa digits as an integer, tracking where the decimal
    // point fell. Beyond 19 significant digits the extra digits cannot affect a
    // float result, so we stop accumulating but keep counting the magnitude.
    std::uint64_t mantissa = 0;
    int exponent = 0;
    int digits = 0;

    while (p < end && is_digit(*p)) {
        if (digits < 19) {
            mantissa = mantissa * 10u + static_cast<std::uint64_t>(*p - '0');
            ++digits;
        } else {
            ++exponent;
        }
        ++p;
    }

    if (p < end && *p == '.') {
        ++p;
        while (p < end && is_digit(*p)) {
            if (digits < 19) {
                mantissa = mantissa * 10u + static_cast<std::uint64_t>(*p - '0');
                ++digits;
                --exponent;
            }
            ++p;
        }
    }

    // Exponent field. Accepts E/e, Fortran D/d, and Fortran's bare-sign form
    // where the 'E' is dropped: 1.234+05.
    if (p < end) {
        bool has_marker = (*p == 'e' || *p == 'E' || *p == 'd' || *p == 'D');
        bool bare_sign = (*p == '+' || *p == '-') && digits > 0;
        if (has_marker || bare_sign) {
            const char* exp_start = p;
            if (has_marker) ++p;

            bool exp_negative = false;
            if (p < end && (*p == '+' || *p == '-')) {
                exp_negative = (*p == '-');
                ++p;
            }

            if (p < end && is_digit(*p)) {
                int exp_value = 0;
                while (p < end && is_digit(*p)) {
                    if (exp_value < 100000) exp_value = exp_value * 10 + (*p - '0');
                    ++p;
                }
                exponent += exp_negative ? -exp_value : exp_value;
            } else {
                p = exp_start;  // not actually an exponent; give the chars back
            }
        }
    }

    *next = p;

    if (mantissa == 0) return negative ? -0.0f : 0.0f;

    double value = scale_pow10(static_cast<double>(mantissa), exponent);
    return static_cast<float>(negative ? -value : value);
}

// Advances past one whitespace-delimited token, eight bytes at a time.
//
// Skipping unwanted columns is a surprisingly large share of the work: reading
// one column out of twenty-four still costs ~90% of what reading all of them
// costs, because every byte has to be examined to find the token boundaries.
// Measured on a 500 MB file, a byte-at-a-time skip ran at ~1.1 GB/s; this runs
// at closer to 2 GB/s.
//
// The test is "byte <= 0x20", which covers space, tab, CR and LF. Every
// character that can appear inside a number -- digits, '.', '-', '+', 'e', 'E',
// 'D' -- is above 0x20, so no numeric byte is ever mistaken for a separator.
//
// The word trick is the classic haszero/hasless idiom: for a 64-bit word x,
//     (x - 0x0101..01 * n) & ~x & 0x8080..80
// is non-zero exactly when some byte of x is less than n. It reads eight bytes
// with one load and needs no shifts per byte.
inline const char* skip_token(const char* p, const char* end) noexcept {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // Locating the first matching byte from the word depends on byte order.
    // Rather than maintain a second version for a case no target of this
    // application has, big-endian builds take the plain loop.
    while (p < end && !is_space(*p)) ++p;
    return p;
#else
    constexpr std::uint64_t kOnes = 0x0101010101010101ull;
    constexpr std::uint64_t kHighBits = 0x8080808080808080ull;
    constexpr std::uint64_t kThreshold = 0x21;  // one past ' '

    while (p + 8 <= end) {
        std::uint64_t word;
        // memcpy rather than a cast: the source is not guaranteed aligned, and
        // a reinterpret_cast here would be undefined behaviour that some
        // compilers do miscompile at -O3.
        std::memcpy(&word, p, sizeof(word));
        const std::uint64_t below = (word - kOnes * kThreshold) & ~word & kHighBits;
        if (below != 0) {
            // Locate the first such byte. Counting trailing zeros gives the bit
            // position; dividing by 8 gives the byte index, little-endian.
#if defined(__GNUC__) || defined(__clang__)
            const int index = __builtin_ctzll(below) >> 3;
#else
            int index = 0;
            std::uint64_t scan = below;
            while ((scan & 0x80ull) == 0) {
                scan >>= 8;
                ++index;
            }
#endif
            return p + index;
        }
        p += 8;
    }

    while (p < end && !is_space(*p)) ++p;
    return p;
#endif
}

// memchr-free substring test over a bounded, non-NUL-terminated line.
inline bool line_contains(const char* line, std::size_t len, const char* needle,
                          std::size_t needle_len) noexcept {
    if (needle_len == 0 || needle_len > len) return false;
    const char first = needle[0];
    const char* p = line;
    const char* limit = line + len - needle_len + 1;
    while (p < limit) {
        if (*p == first) {
            std::size_t i = 1;
            while (i < needle_len && p[i] == needle[i]) ++i;
            if (i == needle_len) return true;
        }
        ++p;
    }
    return false;
}

}  // namespace aoe
