// Author: Paulo R. A. Gilz
//
// Generator for large, realistic AOE result files.
//
// Python cannot write half a gigabyte of formatted text in reasonable time --
// the per-value formatting dominates and it takes minutes per file. This does
// it with a hand-rolled fixed-point formatter into a megabyte-sized buffer,
// which runs at roughly disk speed.
//
//   gen_data out.aoe --size-mb 500 --runs 12 --cols 24
//
// Build:  make gendata

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Channel names that look like a real flight-mechanics run, so the ISO and
// formula features get exercised on plausible data.
const char* kChannelNames[] = {
    "TIME",   "ALPHA",  "BETA",   "MACH",   "NZ",     "NY",
    "ALT",    "THETA",  "PHI",    "PSI",    "VCAS",   "VTAS",
    "P",      "Q",      "R",      "DELTA_E","DELTA_A","DELTA_R",
    "THRUST", "FUEL",   "CL",     "CD",     "MASS",   "XCG",
    "GAMMA",  "CHI",    "TAT",    "SAT",    "QBAR",   "RHO",
    "AOA_L",  "AOA_R",
};
const int kMaxChannels = static_cast<int>(sizeof(kChannelNames) / sizeof(kChannelNames[0]));


// MACH, ALT, MASS and XCG are held constant per run below, so they work as ISO
// parameters when the file is loaded.

// Writes value with 6 decimals into `out`, returning the bytes used.
// snprintf costs roughly 400 ns per value; this is about 25.
int format_fixed(char* out, double value) {
    char* p = out;
    if (value < 0.0) {
        *p++ = '-';
        value = -value;
    }
    if (!(value < 1e15)) {  // NaN or absurd magnitude
        std::memcpy(p, "0.000000", 8);
        return static_cast<int>(p - out) + 8;
    }

    // Round once, then split. Doing it in one 64-bit integer avoids a second
    // rounding step disagreeing with the first.
    const unsigned long long scaled =
        static_cast<unsigned long long>(value * 1000000.0 + 0.5);
    unsigned long long whole = scaled / 1000000ull;
    unsigned int frac = static_cast<unsigned int>(scaled % 1000000ull);

    char digits[24];
    int n = 0;
    if (whole == 0) {
        digits[n++] = '0';
    } else {
        while (whole > 0) {
            digits[n++] = static_cast<char>('0' + (whole % 10));
            whole /= 10;
        }
    }
    while (n > 0) *p++ = digits[--n];

    *p++ = '.';
    for (int i = 5; i >= 0; --i) {
        p[i] = static_cast<char>('0' + (frac % 10));
        frac /= 10;
    }
    p += 6;
    return static_cast<int>(p - out);
}

void usage() {
    std::printf(
        "usage: gen_data OUTPUT [--size-mb N] [--runs N] [--cols N] [--wrap]\n"
        "\n"
        "  --size-mb   target file size in MB (default 500)\n"
        "  --runs      simulation blocks in the file (default 12)\n"
        "  --cols      channels per record (default 24, max %d)\n"
        "  --wrap      split each record over two lines, as Fortran does\n",
        kMaxChannels);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const std::string path = argv[1];
    double target_mb = 500.0;
    int runs = 12;
    int cols = 24;
    bool wrap = false;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--size-mb" && i + 1 < argc) target_mb = std::atof(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (arg == "--cols" && i + 1 < argc) cols = std::atoi(argv[++i]);
        else if (arg == "--wrap") wrap = true;
        else {
            usage();
            return 1;
        }
    }

    if (cols < 2) cols = 2;
    if (cols > kMaxChannels) cols = kMaxChannels;
    if (runs < 1) runs = 1;

    // Each value is " " plus up to ~11 characters; plus one newline per line.
    const int lines_per_record = wrap ? 2 : 1;
    const double bytes_per_row =
        static_cast<double>(cols) * 12.0 + lines_per_record;
    const double target_bytes = target_mb * 1024.0 * 1024.0;
    long long rows_per_run =
        static_cast<long long>(target_bytes / bytes_per_row / runs);
    if (rows_per_run < 1) rows_per_run = 1;

    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) {
        std::printf("cannot open %s for writing\n", path.c_str());
        return 2;
    }

    // A megabyte of buffering turns millions of small writes into a few
    // hundred large ones.
    std::vector<char> buffer(1 << 20);
    std::size_t used = 0;
    auto flush_if_needed = [&](std::size_t needed) {
        if (used + needed > buffer.size()) {
            std::fwrite(buffer.data(), 1, used, file);
            used = 0;
        }
    };

    // Header block.
    {
        std::string header = "*PARAMETRES\n";
        for (int c = 0; c < cols; ++c) {
            header += " '";
            header += kChannelNames[c];
            header += "'";
            if ((c + 1) % 6 == 0) header += "\n";
        }
        if (cols % 6 != 0) header += "\n";
        header += "\n";
        std::fwrite(header.data(), 1, header.size(), file);
    }

    std::vector<double> values(static_cast<std::size_t>(cols));

    for (int run = 0; run < runs; ++run) {
        char marker[128];
        const int marker_length =
            std::snprintf(marker, sizeof(marker), "*TABLEAU : RUN_%03d\n", run + 1);
        flush_if_needed(static_cast<std::size_t>(marker_length));
        std::memcpy(buffer.data() + used, marker, static_cast<std::size_t>(marker_length));
        used += static_cast<std::size_t>(marker_length);

        // Per-run constants, spread over a plausible envelope so grouping the
        // ISO parameters produces something meaningful.
        const double mach = 0.55 + 0.02 * (run % 15);
        const double alt = 5000.0 + 1500.0 * (run % 11);
        const double mass = 180000.0 + 3500.0 * (run % 9);
        const double xcg = 22.0 + 0.75 * (run % 7);

        for (long long row = 0; row < rows_per_run; ++row) {
            const double t = static_cast<double>(row) * 0.01;
            const double phase = t * 0.35 + run * 0.4;

            for (int c = 0; c < cols; ++c) {
                const char* name = kChannelNames[c];
                double v;
                if (std::strcmp(name, "TIME") == 0)        v = t;
                else if (std::strcmp(name, "MACH") == 0)   v = mach;
                else if (std::strcmp(name, "ALT") == 0)    v = alt;
                else if (std::strcmp(name, "MASS") == 0)   v = mass - t * 0.9;
                else if (std::strcmp(name, "XCG") == 0)    v = xcg;
                else if (std::strcmp(name, "ALPHA") == 0)  v = 3.0 + 2.2 * std::sin(phase);
                else if (std::strcmp(name, "NZ") == 0)     v = 1.0 + 0.55 * std::cos(phase * 1.3);
                else {
                    // Everything else is a smooth signal with a per-channel
                    // phase, plus an occasional transient so min/max
                    // decimation has spikes to preserve.
                    v = 10.0 * std::sin(phase + c * 0.31) + 0.4 * std::cos(phase * 4.1 + c);
                    if ((row % 50000) == c * 137 % 50000) v *= 12.0;
                }
                values[static_cast<std::size_t>(c)] = v;
            }

            // 16 bytes of slack per value covers sign, digits and separator.
            flush_if_needed(static_cast<std::size_t>(cols) * 16 + 4);
            char* p = buffer.data() + used;
            for (int c = 0; c < cols; ++c) {
                *p++ = ' ';
                p += format_fixed(p, values[static_cast<std::size_t>(c)]);
                if (wrap && c + 1 == cols / 2) *p++ = '\n';
            }
            *p++ = '\n';
            used = static_cast<std::size_t>(p - buffer.data());
        }
    }

    const char* trailer = "*FIN\n";
    flush_if_needed(6);
    std::memcpy(buffer.data() + used, trailer, 5);
    used += 5;
    std::fwrite(buffer.data(), 1, used, file);

    const long long size = std::ftell(file);
    std::fclose(file);

    std::printf("%s  %.1f MB  %d runs x %lld rows x %d channels\n", path.c_str(),
                static_cast<double>(size) / (1024.0 * 1024.0), runs, rows_per_run, cols);
    return 0;
}
