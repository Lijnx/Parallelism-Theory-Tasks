#include <fstream>
#include <vector>
#include <chrono>
#include <cstddef>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/global_control.h>


void dgemv_tbb(
    const std::vector<double>& a,
    const std::vector<double>& b,
    std::vector<double>& c,
    std::size_t m,
    std::size_t n
) {
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<std::size_t>(0, m),
        [&](const oneapi::tbb::blocked_range<std::size_t>& range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                const double* row = &a[i * n];

                double sum = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    sum += row[j] * b[j];
                }

                c[i] = sum;
            }
        }
    );
}


double run_parallel(std::size_t m, std::size_t n) {
    std::vector<double> a(m * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<std::size_t>(0, m),
        [&](const oneapi::tbb::blocked_range<std::size_t>& range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                double* row = &a[i * n];

                for (std::size_t j = 0; j < n; ++j) {
                    row[j] = static_cast<double>(i + j);
                }

                c[i] = 0.0;
            }
        }
    );

    for (std::size_t j = 0; j < n; ++j) {
        b[j] = static_cast<double>(j);
    }

    const auto start = std::chrono::steady_clock::now();

    dgemv_tbb(a, b, c, m, n);

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds(end - start);

    return elapsed_seconds.count();
}


int main(int argc, char** argv) {
    const int tests_num = 100;

    std::vector<int> test_threads{1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<int> data_size{20000, 40000};

    std::ofstream fout("dgemv_time.csv");

    fout << "data_size, "
         << "threads, "
         << "time" << std::endl;

    for (int ds : data_size) {
        for (int nthreads : test_threads) {

            oneapi::tbb::global_control thread_limit(
                oneapi::tbb::global_control::max_allowed_parallelism,
                nthreads
            );

            for (int t = 0; t < tests_num; ++t) {
                double time = run_parallel(
                    static_cast<std::size_t>(ds),
                    static_cast<std::size_t>(ds)
                );

                fout << ds       << ", "
                     << nthreads << ", "
                     << time     << std::endl;
            }
        }
    }

    fout.close();

    return 0;
}