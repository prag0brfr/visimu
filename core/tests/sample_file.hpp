// Author: Paulo R. A. Gilz
// Generates AOE-format files for tests and benchmarks.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace aoe_test {

// Writes a file with `sims` simulation blocks of `rows` rows each, over the
// columns TIME, ALPHA, MACH, NZ (MACH is constant per block, so it works as an
// ISO parameter). Returns false if the file could not be written.
inline bool write_sample(const std::string& path, int sims, int rows,
                         bool wrap_records = false) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    std::fprintf(f, "*PARAMETRES\n");
    std::fprintf(f, " 'TIME'   'ALPHA'   'MACH'   'NZ'\n");
    std::fprintf(f, "\n");

    for (int s = 0; s < sims; ++s) {
        std::fprintf(f, "*TABLEAU : SIM_%03d\n", s + 1);
        const double mach = 0.60 + 0.05 * s;
        for (int r = 0; r < rows; ++r) {
            const double t = r * 0.02;
            const double alpha = 2.0 + std::sin(t) * (1.0 + 0.1 * s);
            const double nz = 1.0 + 0.25 * std::cos(t * 0.5);
            if (wrap_records) {
                // Same record split across two physical lines, the Fortran
                // wrapping the parser has to cope with.
                std::fprintf(f, " %.6f %.6f\n %.6f %.6f\n", t, alpha, mach, nz);
            } else {
                std::fprintf(f, " %.6f %.6f %.6f %.6f\n", t, alpha, mach, nz);
            }
        }
    }
    std::fprintf(f, "*FIN\n");
    std::fclose(f);
    return true;
}

// The other *PARAMETRES dialect: one parameter per line, prefixed by its
// ordinal and followed by its unit, with an *ENTITY block of free text before
// the first table. Same channels and the same numbers as write_sample(), so
// the two can be compared value for value.
inline bool write_sample_indexed(const std::string& path, int sims, int rows,
                                 bool wrap_records = false) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    static const char* kNames[] = {"TIME", "ALPHA", "MACH", "NZ"};
    static const char* kUnits[] = {"s", "deg", "", ""};

    std::fprintf(f, "*PARAMETRES\t\t\t\n");
    for (int i = 0; i < 4; ++i)
        std::fprintf(f, "\t%d\t'%s'\t \t'%s'\n", i + 1, kNames[i], kUnits[i]);
    std::fprintf(f, "\n");

    std::fprintf(f, "*ENTITY\t\t\t\n");
    std::fprintf(f, "\n");
    std::fprintf(f, "Free text with digits 123 and punctuation !@#\n");
    std::fprintf(f, "\n");

    for (int s = 0; s < sims; ++s) {
        std::fprintf(f, "*TABLEAU NUM SIMU: %d.2.0\n", s + 1);
        const double mach = 0.60 + 0.05 * s;
        for (int r = 0; r < rows; ++r) {
            const double t = r * 0.02;
            const double alpha = 2.0 + std::sin(t) * (1.0 + 0.1 * s);
            const double nz = 1.0 + 0.25 * std::cos(t * 0.5);
            if (wrap_records) {
                std::fprintf(f, " %.6f %.6f\n %.6f %.6f\n", t, alpha, mach, nz);
            } else {
                std::fprintf(f, " %.6f %.6f %.6f %.6f\n", t, alpha, mach, nz);
            }
        }
    }
    std::fprintf(f, "*FIN\n");
    std::fclose(f);
    return true;
}

}  // namespace aoe_test
