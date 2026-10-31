// Author: Paulo R. A. Gilz
// Parser throughput benchmark. Writes a large synthetic file, then times a cold
// full-column load, so the number reported is the one users actually wait for.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "aoe/session.hpp"
#include "aoe/thread_pool.hpp"
#include "sample_file.hpp"

namespace {

double seconds_since(const std::chrono::steady_clock::time_point& start) {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

void run(const std::string& path, int sims, int rows) {
    const auto write_start = std::chrono::steady_clock::now();
    if (!aoe_test::write_sample(path, sims, rows)) {
        std::printf("could not write %s\n", path.c_str());
        return;
    }
    const double write_time = seconds_since(write_start);

    std::FILE* f = std::fopen(path.c_str(), "rb");
    std::fseek(f, 0, SEEK_END);
    const double megabytes = static_cast<double>(std::ftell(f)) / (1024.0 * 1024.0);
    std::fclose(f);

    std::printf("\n%d runs x %d rows  (%.1f MB, written in %.2f s)\n", sims, rows,
                megabytes, write_time);

    aoe::Session session;
    const auto index_start = std::chrono::steady_clock::now();
    const int file = session.add_file(path);
    const double index_time = seconds_since(index_start);
    if (file < 0) {
        std::printf("  failed to open\n");
        return;
    }
    std::printf("  index scan        %7.3f s   %7.1f MB/s\n", index_time,
                megabytes / index_time);

    // Cold load of every column, which is the worst case the UI can ask for.
    const auto load_start = std::chrono::steady_clock::now();
    session.preload(file, {}, {"TIME", "ALPHA", "MACH", "NZ"});
    const double load_time = seconds_since(load_start);

    std::size_t samples = 0;
    for (int s = 0; s < sims; ++s) {
        std::size_t n = 0;
        session.channel(file, s, "TIME", &n);
        samples += n * 4;
    }

    std::printf("  parse 4 columns   %7.3f s   %7.1f MB/s   %.1f M samples/s\n",
                load_time, megabytes / load_time,
                static_cast<double>(samples) / load_time / 1e6);

    // Warm access must be effectively free -- this is what the UI hits on every
    // repaint, and any real cost here shows up as dropped frames.
    const auto warm_start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 1000; ++repeat)
        for (int s = 0; s < sims; ++s) {
            std::size_t n = 0;
            session.channel(file, s, "ALPHA", &n);
        }
    const double warm_time = seconds_since(warm_start);
    std::printf("  cached lookup     %7.1f ns per call\n",
                warm_time / (1000.0 * sims) * 1e9);

    std::remove(path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("visimu parser benchmark  (%d worker threads)\n",
                static_cast<int>(aoe::ThreadPool::global().size()) + 1);

    const std::string path = (argc > 1) ? argv[1] : "aoe_bench_tmp.dat";

    run(path, 1, 500000);   // one big table: exercises chunked parallel parsing
    run(path, 8, 100000);   // several runs: exercises per-simulation parallelism
    run(path, 64, 20000);   // many small runs

    return 0;
}
