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
    double
);


void slae_single_block(
    const std::vector<double>& a, 
    std::vector<double>& x, 
    const std::vector<double>& b, 
    int n,
    double t, 
    double e
) {

    std::vector<double> x_new(n);
    double b_norm = 0.0;
    double norm_sq = 0.0;
    bool done = false;

    #pragma omp parallel
    {

        #pragma omp for reduction(+:b_norm)
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

            #pragma omp for reduction(+:norm_sq)
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

            #pragma omp for
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


void slae_multiple_blocks(
    const std::vector<double>& a, 
    std::vector<double>& x, 
    const std::vector<double>& b, 
    int n,
    double t, 
    double e
) {

    std::vector<double> x_new(n);
    double b_norm = 0.0;
    bool done = false;

    #pragma omp parallel for reduction(+:b_norm)
    for (int i = 0; i < n; ++i) {
        b_norm += b[i] * b[i];
    }
    b_norm = std::sqrt(b_norm);

    while (!done) {
        double norm_sq = 0.0;

        #pragma omp parallel for reduction(+:norm_sq)
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

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            x[i] = x_new[i];
        }

        double norm = std::sqrt(norm_sq);
        done = (norm / b_norm < e);
    }
}


double run_parallel(int n, Slae slae) {

    std::vector<double> a(n * n);
    std::vector<double> x(n);
    std::vector<double> b(n);
    double t = 1.0e-5;
    double e = 1.0e-5;

    #pragma omp parallel for
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

    slae(a, x, b, n, t, e);

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds(end - start);

    return elapsed_seconds.count();
}


void benchmark(
    int data_size, 
    int tests_num, 
    const std::vector<int>& test_threads, 
    Slae slae, 
    std::string output_file_name
) {

    std::ofstream fout(output_file_name);
    fout << "threads, "
         << "time"      << std::endl;

    for (int nthreads : test_threads) {
        omp_set_num_threads(nthreads);
        for (int t = 0; t < tests_num; ++t) {
            double time = run_parallel(data_size, slae);
            fout << nthreads << ", "
                 << time     << std::endl;
        }       
    }

    fout.close();
}


int main(int argc, char** argv) {

    const int data_size = 10000;
    const int tests_num = 100;
    std::vector<int> test_threads{1,2,4,7,8,16,20,40};

    benchmark(data_size, tests_num, test_threads, slae_single_block, "slae_single_block.csv");
    benchmark(data_size, tests_num, test_threads, slae_multiple_blocks, "slae_multiple_blocks.csv");

    return 0;
}
