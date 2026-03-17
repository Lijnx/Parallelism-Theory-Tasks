#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

void dgemv_omp(
    const std::vector<double>& a, 
    const std::vector<double>& b, 
    std::vector<double>& c, 
    int m, int n
) {

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; ++i) {
            const double* row = &a[i * n];
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += row[j] * b[j];
            }
            c[i] = sum;
        }
    }
}

double run_parallel(int m, int n) {

    std::vector<double> a(m * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        
        for (int i = lb; i <= ub; ++i) {
            double* row = &a[i * n];
            for (int j = 0; j < n; ++j) {
                row[j] = i + j;
            }
            c[i] = 0.0;
        }
    }
    for (int j = 0; j < n; ++j) {
        b[j] = j;
    }

    const auto start{std::chrono::steady_clock::now()};

    dgemv_omp(a, b, c, m, n);

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds(end - start);

    return elapsed_seconds.count();
}

int main(int argc, char** argv) {

    const int tests_num = 100; // for each data_size
    std::vector<int> test_threads{1,2,4,7,8,16,20,40};
    std::vector<int> data_size{20000, 40000};

    std::ofstream fout("dgemv_time.csv");
    fout << "data_size, "
         << "threads, "
         << "time" << std::endl;

    for (int ds : data_size) {

        for (int nthreads : test_threads) {

            omp_set_num_threads(nthreads);
            for (int t = 0; t < tests_num; ++t) {

                double time = run_parallel(ds, ds);
                fout << ds       << ", "
                     << nthreads << ", "
                     << time     << std::endl;

            }
        }
    }

    fout.close();

    return 0;
}