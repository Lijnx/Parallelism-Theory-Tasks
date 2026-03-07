#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <omp.h>

size_t idx(size_t i, size_t j, size_t cols) {
    return i * cols + j;
}

void dgemv_omp(
    const std::vector<double>& a, 
    const std::vector<double>& b, 
    std::vector<double>& c, 
    size_t m, size_t n, 
    int nthreads = omp_get_num_threads()
) {

    #pragma omp parallel num_threads(nthreads)
    {
        // int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        for (size_t i = lb; i <= ub; ++i) {
            c[i] = 0.0;
            for (size_t j = 0; j < n; ++j) {
                c[i] += a[idx(i,j,n)] * b[j];
            }
        }
    }
}

double run_parallel(size_t m, size_t n, int thread_num) {

    std::vector<double> a(m * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        for (size_t i = lb; i <= ub; ++i) {
            for (size_t j = 0; j < n; ++j) {
                a[idx(i,j,n)] = i + j;
            }
            b[i] = i;
        }
    }

    const auto start{std::chrono::steady_clock::now()};

    dgemv_omp(a, b, c, m, n, thread_num);

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds(end - start);

    return elapsed_seconds.count();
}

double benchmark(size_t data_size, int tests_num, int threads_num) {

    std::vector<double> time(tests_num);
    for (size_t i = 0; i < tests_num; ++i) {
        double t = run_parallel(data_size, data_size, threads_num);
        time.push_back(t);
    }
    std::sort(time.begin(), time.end());
    time.erase(time.begin(), time.begin() + tests_num/10); // delete lower 10%
    time.erase(time.end() - tests_num/10, time.end());; // delete higher 10%

    double sum = std::accumulate(time.begin(), time.end(), 0.0);
    return sum / time.size();
}

int main(int argc, char** argv) {

    const int tests_num = 10; // for each data_size
    std::vector<int> test_threads{1,2,4,7,8,16,20,40}; // 1 - REQUIRED!

    std::ofstream fout("dgemv_time.csv");
    fout << std::fixed << std::setprecision(2);

    fout << "data_size, "
         << "threads, "
         << "time, "
         << "acceleration" << std::endl;

    double time1_20k, time1_40k;
    for (int nthreads : test_threads) {
        double time;
        size_t data_size;

        data_size = 20000;
        time = benchmark(data_size, tests_num, nthreads);
        time1_20k = (nthreads == 1) ? time : time1_20k;
        fout << data_size      << ", "
             << nthreads       << ", "
             << time           << ", "
             << time1_20k/time << std::endl;


        data_size = 40000;
        time = benchmark(40000, tests_num, nthreads);
        time1_40k = (nthreads == 1) ? time : time1_40k;
        fout << data_size      << ", "
             << nthreads       << ", "
             << time           << ", "
             << time1_40k/time << std::endl;
    }

    fout.close();

    return 0;
}