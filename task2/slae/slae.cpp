
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>

size_t idx(size_t i, size_t j, size_t cols) {
    return i * cols + j;
}

void slae_iter_omp(
    const std::vector<double>& a, 
    std::vector<double>& x, 
    const std::vector<double>& b, 
    size_t n,
    double t, double e,
    int nthreads = omp_get_num_threads()
) {

    double b_norm = 0.0;
    #pragma omp parallel num_threads(nthreads)
    {
        // int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        double norm_chunk = 0.0;
        for (size_t i = lb; i <= ub; ++i) {
            norm_chunk += b[i] * b[i];
        }

        #pragma omp atomic
        b_norm += norm_chunk;
    }
    b_norm = sqrt(b_norm);

    double norm;
    do {
        norm = 0.0;
        #pragma omp parallel num_threads(nthreads)
        {
            // int nthreads = omp_get_num_threads();
            int threadid = omp_get_thread_num();
            int items_per_thread = n / nthreads;
            int lb = threadid * items_per_thread;
            int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

            double norm_chunk = 0.0;
            for (size_t i = lb; i <= ub; ++i) {
                double Ax_i = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    Ax_i += a[idx(i,j,n)] * x[j];
                }
                x[i] -= t * (Ax_i - b[i]);
                norm_chunk += (Ax_i - b[i]) * (Ax_i - b[i]);
            }

            #pragma omp atomic
            norm += norm_chunk;
        }
    } while (norm / b_norm >= e);

}

double run_parallel(size_t n, int threads_num) {

    std::vector<double> a(n * n);
    std::vector<double> x(n);
    std::vector<double> b(n);
    double t = 0.01;
    double e = 1.0e-5;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        for (size_t i = lb; i <= ub; ++i) {
            for (size_t j = 0; j < n; ++j) {
                a[idx(i,j,n)] = 1.0;
            }
            a[idx(i,i,n)] = 2.0;
            b[i] = n + 1.0;
            x[i] = 0.0;
        }
    }

    const auto start{std::chrono::steady_clock::now()};

    slae_iter_omp(a, x, b, n, t, e, threads_num);

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds(end - start);

    return elapsed_seconds.count();
}

double benchmark(size_t n, int tests_num, int threads_num) {

    std::vector<double> time(tests_num);
    for (size_t i = 0; i < tests_num; ++i) {
        double t = run_parallel(n, threads_num);
        time.push_back(t);
    }
    std::sort(time.begin(), time.end());
    time.erase(time.begin(), time.begin() + tests_num/10); // delete lower 10%
    time.erase(time.end() - tests_num/10, time.end());; // delete higher 10%

    double sum = std::accumulate(time.begin(), time.end(), 0.0);
    return sum / time.size();
}

int main(int argc, char** argv) {

    const int n = 1000;
    const int tests_num = 10; // for each thread
    std::vector<int> test_threads{1,2,4,7,8,16,20,40}; // 1 - REQUIRED!

    std::ofstream fout("slae_time.csv");
    fout << std::fixed << std::setprecision(2);

    fout << "threads, "
         << "time, "
         << "acceleration" << std::endl;

    double time1;
    for (int nthreads : test_threads) {
        double time;

        time = benchmark(n, tests_num, nthreads);
        time1 = (nthreads == 1) ? time : time1;
        fout << nthreads       << ", "
             << time           << ", "
             << time1/time << std::endl;
    }

    fout.close();

    return 0;
}