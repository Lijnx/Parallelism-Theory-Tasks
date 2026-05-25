#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#ifdef _OPENACC
#include <openacc.h>
#endif

namespace fs = std::filesystem;
namespace po = boost::program_options;

struct Options {
    int size = 512;
    double eps = 1e-6;
    int max_iters = 1000000;
    int runs = 3;
    std::vector<int> bench_sizes = {128, 256, 512, 1024};
    std::vector<std::string> modes = {"serial", "openacc"};
    std::string mode = "serial";
    std::string device = "default";
    std::string csv = "results/benchmark.csv";
    std::string save = "";
    bool bench = false;
    bool help = false;
};

struct SolveResult {
    double seconds = 0.0;
    int iterations = 0;
    double error = 0.0;
    std::vector<double> grid;
};

static inline std::size_t at(int row, int col, int n) {
    return static_cast<std::size_t>(row) * n + col;
}

static inline double lerp(double a, double b, double t) {
    return a + t * (b - a);
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static void validate_size(int n) {
    if (n < 3) {
        throw std::invalid_argument("Grid size must be at least 3.");
    }
}

static void init_grid(std::vector<double>& grid, int n) {
    validate_size(n);
    grid.assign(static_cast<std::size_t>(n) * n, 0.0);

    const double top_left = 10.0;
    const double top_right = 20.0;
    const double bottom_right = 30.0;
    const double bottom_left = 20.0;

    for (int col = 0; col < n; ++col) {
        const double t = col / static_cast<double>(n - 1);
        grid[at(0, col, n)] = lerp(top_left, top_right, t);
        grid[at(n - 1, col, n)] = lerp(bottom_left, bottom_right, t);
    }

    for (int row = 0; row < n; ++row) {
        const double t = row / static_cast<double>(n - 1);
        grid[at(row, 0, n)] = lerp(top_left, bottom_left, t);
        grid[at(row, n - 1, n)] = lerp(top_right, bottom_right, t);
    }
}

static void ensure_parent_dir(const std::string& path) {
    const fs::path p(path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
}

static std::string device_label(const std::string& device) {
    const std::string d = lower(device);
    if (d == "cpu" || d == "host") {
        return "cpu";
    }
    if (d == "gpu" || d == "nvidia") {
        return "gpu";
    }
    if (d == "multicore") {
        return "cpu-multicore";
    }
    return "default";
}

static void configure_openacc_device(const std::string& device) {
    const std::string d = lower(device);
#ifdef _OPENACC
    if (d == "cpu" || d == "host") {
        acc_set_device_type(acc_device_host);
    } else if (d == "gpu" || d == "nvidia") {
        acc_set_device_type(acc_device_nvidia);
    }
#else
    (void)d;
#endif
}

static SolveResult solve_serial(int n, double eps, int max_iters, bool keep_grid) {
    std::vector<double> grid;
    std::vector<double> next;
    init_grid(grid, n);
    init_grid(next, n);

    int iter = 0;
    double err = 1.0;
    const auto t0 = std::chrono::steady_clock::now();

    while (iter < max_iters && err > eps) {
        err = 0.0;
        for (int row = 1; row < n - 1; ++row) {
            for (int col = 1; col < n - 1; ++col) {
                const double value =
                    0.25 * (grid[at(row - 1, col, n)] + grid[at(row + 1, col, n)] +
                            grid[at(row, col - 1, n)] + grid[at(row, col + 1, n)]);
                next[at(row, col, n)] = value;
                err = std::max(err, std::fabs(value - grid[at(row, col, n)]));
            }
        }

        for (int row = 1; row < n - 1; ++row) {
            for (int col = 1; col < n - 1; ++col) {
                grid[at(row, col, n)] = next[at(row, col, n)];
            }
        }
        ++iter;
    }

    const auto t1 = std::chrono::steady_clock::now();
    SolveResult result;
    result.seconds = std::chrono::duration<double>(t1 - t0).count();
    result.iterations = iter;
    result.error = err;
    if (keep_grid) {
        result.grid = std::move(grid);
    }
    return result;
}

static SolveResult solve_openacc(int n, double eps, int max_iters, const std::string& device, bool keep_grid) {
    configure_openacc_device(device);

    std::vector<double> grid;
    std::vector<double> next;
    init_grid(grid, n);
    init_grid(next, n);

    double* a = grid.data();
    double* b = next.data();
    [[maybe_unused]] const std::size_t sz = static_cast<std::size_t>(n) * n;

    int iter = 0;
    double err = 1.0;
    const auto t0 = std::chrono::steady_clock::now();

    #pragma acc data copy(a[0:sz], b[0:sz])
    {
        while (iter < max_iters && err > eps) {
            err = 0.0;

            #pragma acc parallel loop collapse(2) reduction(max:err)
            for (int row = 1; row < n - 1; ++row) {
                for (int col = 1; col < n - 1; ++col) {
                    const std::size_t idx = static_cast<std::size_t>(row) * n + col;
                    const double value =
                        0.25 * (a[idx - n] + a[idx + n] + a[idx - 1] + a[idx + 1]);
                    b[idx] = value;
                    const double diff = std::fabs(value - a[idx]);
                    if (diff > err) {
                        err = diff;
                    }
                }
            }

            #pragma acc parallel loop collapse(2)
            for (int row = 1; row < n - 1; ++row) {
                for (int col = 1; col < n - 1; ++col) {
                    const std::size_t idx = static_cast<std::size_t>(row) * n + col;
                    a[idx] = b[idx];
                }
            }

            ++iter;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    SolveResult result;
    result.seconds = std::chrono::duration<double>(t1 - t0).count();
    result.iterations = iter;
    result.error = err;
    if (keep_grid) {
        result.grid = std::move(grid);
    }
    return result;
}

static SolveResult solve(const std::string& mode, int n, double eps, int max_iters,
                         const std::string& device, bool keep_grid) {
    const std::string m = lower(mode);
    if (m == "serial" || m == "baseline") {
        return solve_serial(n, eps, max_iters, keep_grid);
    }
    if (m == "openacc" || m == "acc") {
        return solve_openacc(n, eps, max_iters, device, keep_grid);
    }
    throw std::invalid_argument("Unknown mode: " + mode);
}

static void save_grid(const std::string& filename, int n, const std::vector<double>& grid) {
    ensure_parent_dir(filename);
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot open output grid file: " + filename);
    }

    out << std::fixed << std::setprecision(6);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            out << grid[at(row, col, n)];
            if (col + 1 < n) {
                out << ',';
            }
        }
        out << '\n';
    }
}

static po::options_description make_description(Options& opt) {
    po::options_description desc("Heat 2D solver options");
    desc.add_options()
        ("help,h", "Show help")
        ("mode", po::value<std::string>(&opt.mode)->default_value(opt.mode),
            "Implementation for a single run: serial or openacc")
        ("device", po::value<std::string>(&opt.device)->default_value(opt.device),
            "OpenACC device label/runtime target: default, cpu, gpu or multicore")
        ("size,n", po::value<int>(&opt.size)->default_value(opt.size),
            "Grid size NxN")
        ("eps,e", po::value<double>(&opt.eps)->default_value(opt.eps),
            "Convergence threshold")
        ("iters,i", po::value<int>(&opt.max_iters)->default_value(opt.max_iters),
            "Max iterations")
        ("runs,r", po::value<int>(&opt.runs)->default_value(opt.runs),
            "Runs per size in benchmark mode")
        ("sizes", po::value<std::vector<int>>(&opt.bench_sizes)->multitoken(),
            "Benchmark sizes, for example: --sizes 128 256 512")
        ("modes", po::value<std::vector<std::string>>(&opt.modes)->multitoken(),
            "Benchmark implementations, for example: --modes serial openacc")
        ("bench", po::bool_switch(&opt.bench),
            "Run benchmark and write CSV")
        ("csv", po::value<std::string>(&opt.csv)->default_value(opt.csv),
            "Benchmark CSV path")
        ("save", po::value<std::string>(&opt.save),
            "Save resulting matrix as CSV-like text");
    return desc;
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    const auto desc = make_description(opt);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    opt.help = vm.count("help") > 0;
    opt.mode = lower(opt.mode);
    opt.device = lower(opt.device);
    for (std::string& mode : opt.modes) {
        mode = lower(mode);
    }

    validate_size(opt.size);
    for (int s : opt.bench_sizes) {
        validate_size(s);
    }
    if (opt.runs < 1) {
        throw std::invalid_argument("Runs must be positive.");
    }
    if (opt.max_iters < 1) {
        throw std::invalid_argument("Max iterations must be positive.");
    }
    if (opt.eps <= 0.0) {
        throw std::invalid_argument("Epsilon must be positive.");
    }
    return opt;
}

static void write_benchmark(const Options& opt) {
    ensure_parent_dir(opt.csv);
    std::ofstream csv(opt.csv);
    if (!csv) {
        throw std::runtime_error("Cannot open CSV file: " + opt.csv);
    }

    csv << "mode,device,size,run,time_sec,iterations,error,eps,max_iters\n";
    for (const std::string& mode : opt.modes) {
        for (int n : opt.bench_sizes) {
            for (int run = 1; run <= opt.runs; ++run) {
                const SolveResult result = solve(mode, n, opt.eps, opt.max_iters, opt.device, false);
                csv << lower(mode) << ','
                    << (lower(mode) == "openacc" ? device_label(opt.device) : "cpu-onecore") << ','
                    << n << ','
                    << run << ','
                    << std::setprecision(10) << result.seconds << ','
                    << result.iterations << ','
                    << std::scientific << std::setprecision(6) << result.error << std::defaultfloat << ','
                    << opt.eps << ','
                    << opt.max_iters << '\n';

                std::cout << "mode=" << mode
                          << " device=" << (lower(mode) == "openacc" ? device_label(opt.device) : "cpu-onecore")
                          << " size=" << n
                          << " run=" << run
                          << " time=" << std::fixed << std::setprecision(4) << result.seconds << "s"
                          << " iters=" << result.iterations
                          << " err=" << std::scientific << std::setprecision(3) << result.error
                          << std::defaultfloat << '\n';
            }
        }
    }
    std::cout << "Results written to " << opt.csv << '\n';
}

int main(int argc, char** argv) {
    try {
        Options defaults;
        const po::options_description desc = make_description(defaults);
        const Options opt = parse_args(argc, argv);
        if (opt.help) {
            std::cout << "Usage: " << argv[0] << " [options]\n\n" << desc << '\n';
            return 0;
        }

        if (opt.bench) {
            write_benchmark(opt);
            return 0;
        }

        const bool keep_grid = !opt.save.empty();
        const SolveResult result = solve(opt.mode, opt.size, opt.eps, opt.max_iters, opt.device, keep_grid);

        std::cout << "mode=" << opt.mode
                  << " device=" << (lower(opt.mode) == "openacc" ? device_label(opt.device) : "cpu-onecore")
                  << " size=" << opt.size
                  << " time=" << std::fixed << std::setprecision(6) << result.seconds << "s"
                  << " iterations=" << result.iterations
                  << " error=" << std::scientific << std::setprecision(3) << result.error
                  << std::defaultfloat << '\n';

        if (keep_grid) {
            save_grid(opt.save, opt.size, result.grid);
            std::cout << "Grid saved to " << opt.save << '\n';
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        std::cerr << "Run with --help for usage.\n";
        return 1;
    }
    return 0;
}
