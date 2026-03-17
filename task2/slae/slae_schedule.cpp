#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>


using Slae = void (*)(
    const std::vector<double>&, 
    std::vector<double>&, 
    const std::vector<double>&, 
    int, 
    double, 
    double,
    int
);


void slae_static(
    const std::vector<double>& a, 
    std::vector<double>& x, 
    const std::vector<double>& b, 
    int n,
    double t, 
    double e,
    int k
) {

    std::vector<double> x_new(n);
    double b_norm = 0.0;
    double norm_sq = 0.0;
    bool done = false;

    #pragma omp parallel
    {

        #pragma omp for schedule(static, k) reduction(+:b_norm)
        for (int i = 0; i < n; ++i) {
            b_norm += b[i] * b[i];
        }

        #pragma omp single
        {
            b_norm = std::sqrt(b_norm);
        }
        #pragma omp barrier

        while (!done) {

            #pragma omp single
            {
                norm_sq = 0.0;
            }

            #pragma omp for schedule(static, k) reduction(+:norm_sq)
            for (int i = 0; i < n; ++i) {

                const double* row = &a[i * n];
                double Ax_i = 0.0;
                for (int j = 0; j < n; ++j) {
                    Ax_i += row[j] * x[j];
                }

                double ri = Ax_i - b[i];
                x_new[i] = x[i] - t * ri;
                norm_sq += ri * ri;
            }

            #pragma omp for schedule(static, k)
            for (int i = 0; i < n; ++i) {
                x[i] = x_new[i];
            }

            #pragma omp single
            {
                double norm = std::sqrt(norm_sq);
                done = (norm / b_norm < e);
            }
            #pragma omp barrier
        }
    }
}


void slae_dynamic(
    const std::vector<double>& a, 
    std::vector<double>& x, 
    const std::vector<double>& b, 
    int n,
    double t, 
    double e,
    int k
) {

    std::vector<double> x_new(n);
    double b_norm = 0.0;
    double norm_sq = 0.0;
    bool done = false;

    #pragma omp parallel
    {

        #pragma omp for schedule(dynamic, k) reduction(+:b_norm)
        for (int i = 0; i < n; ++i) {
            b_norm += b[i] * b[i];
        }

        #pragma omp single
        {
            b_norm = std::sqrt(b_norm);
        }
        #pragma omp barrier

        while (!done) {

            #pragma omp single
            {
                norm_sq = 0.0;
            }

            #pragma omp for schedule(dynamic, k) reduction(+:norm_sq)
            for (int i = 0; i < n; ++i) {

                const double* row = &a[i * n];
                double Ax_i = 0.0;
                for (int j = 0; j < n; ++j) {
                    Ax_i += row[j] * x[j];
                }

                double ri = Ax_i - b[i];
                x_new[i] = x[i] - t * ri;
                norm_sq += ri * ri;
            }

            #pragma omp for schedule(dynamic, k)
            for (int i = 0; i < n; ++i) {
                x[i] = x_new[i];
            }

            #pragma omp single
            {
                double norm = std::sqrt(norm_sq);
                done = (norm / b_norm < e);
            }
            #pragma omp barrier
        }
    }
}


void slae_guided(
    const std::vector<double>& a, 
    std::vector<double>& x, 
    const std::vector<double>& b, 
    int n,
    double t, 
    double e,
    int k
) {

    std::vector<double> x_new(n);
    double b_norm = 0.0;
    double norm_sq = 0.0;
    bool done = false;

    #pragma omp parallel
    {

        #pragma omp for schedule(guided, k) reduction(+:b_norm)
        for (int i = 0; i < n; ++i) {
            b_norm += b[i] * b[i];
        }

        #pragma omp single
        {
            b_norm = std::sqrt(b_norm);
        }
        #pragma omp barrier

        while (!done) {

            #pragma omp single
            {
                norm_sq = 0.0;
            }

            #pragma omp for schedule(guided, k) reduction(+:norm_sq)
            for (int i = 0; i < n; ++i) {

                const double* row = &a[i * n];
                double Ax_i = 0.0;
                for (int j = 0; j < n; ++j) {
                    Ax_i += row[j] * x[j];
                }

                double ri = Ax_i - b[i];
                x_new[i] = x[i] - t * ri;
                norm_sq += ri * ri;
            }

            #pragma omp for schedule(guided, k)
            for (int i = 0; i < n; ++i) {
                x[i] = x_new[i];
            }

            #pragma omp single
            {
                double norm = std::sqrt(norm_sq);
                done = (norm / b_norm < e);
            }
            #pragma omp barrier
        }
    }
}


double run_parallel(int n, int k, Slae slae) {

    std::vector<double> a(n * n);
    std::vector<double> x(n);
    std::vector<double> b(n);
    double t = 1.0e-5;
    double e = 1.0e-5;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double* row = &a[i * n];
        for (int j = 0; j < n; ++j) {
            row[j] = 1.0;
        }
        row[i] = 2.0;
        b[i] = n + 1.0;
        x[i] = 0.0;
    }
    
    const auto start{std::chrono::steady_clock::now()};

    slae(a, x, b, n, t, e, k);

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds(end - start);

    return elapsed_seconds.count();
}


void benchmark(
    int data_size, 
    int tests_num, 
    const std::vector<int>& test_threads, 
    std::vector<int>& test_chunks,
    Slae slae,
    std::string output_file_name
) {

    std::ofstream fout(output_file_name);
    fout << "threads, "
         << "chunks, "
         << "time"      << std::endl;

    for (int nthreads : test_threads) {
        omp_set_num_threads(nthreads);

        for (int chunk_size : test_chunks) {

            for (int t = 0; t < tests_num; ++t) {
                double time = run_parallel(data_size, chunk_size, slae);
                fout << nthreads   << ", "
                     << chunk_size << ", "
                     << time       << std::endl;
            }     
        }  
    }

    fout.close();
}


int main(int argc, char** argv) {

    const int data_size = 10000;
    const int tests_num = 100;
    std::vector<int> test_threads{1,2,4,7,8,16,20,40};
    std::vector<int> test_chunks{1,10,100,1000};

    benchmark(data_size, tests_num, test_threads, test_chunks, slae_static, "slae_static.csv");
    benchmark(data_size, tests_num, test_threads, test_chunks, slae_dynamic, "slae_dynamic.csv");
    benchmark(data_size, tests_num, test_threads, test_chunks, slae_guided, "slae_guided.csv");

    return 0;
}
