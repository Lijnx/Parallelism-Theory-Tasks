#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <cmath>

double func(double x) {
    return exp(-x * x);
}

double integrate_omp(
    double (*func)(double), 
    double a, double b,
    int nsteps = 40'000'000, 
    int nthreads = omp_get_num_threads()
) {

    double h = (b - a) / nsteps;
    double sum = 0.0;
    #pragma omp parallel num_threads(nthreads)
    {
        int threadid = omp_get_thread_num();
        int items_per_thread = nsteps / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (nsteps - 1) : (lb + items_per_thread - 1);
        
        double sumloc = 0.0;
        for (int i = lb; i <= ub; i++) {
            sumloc += func(a + h * (i + 0.5));
        }
        
        #pragma omp atomic
        sum += sumloc;
    }
    sum *= h;
    return sum;
}

double run_parallel(int thread_num) {

    const double a = -4.0;
    const double b = 4.0;
    const int nsteps = 40000000;

    const auto start{std::chrono::steady_clock::now()};

    integrate_omp(func, a, b, nsteps, thread_num);

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds(end - start);

    return elapsed_seconds.count();
}

double benchmark(int tests_num, int threads_num) {

    std::vector<double> time(tests_num);
    for (size_t i = 0; i < tests_num; ++i) {
        double t = run_parallel(threads_num);
        time.push_back(t);
    }
    std::sort(time.begin(), time.end());
    time.erase(time.begin(), time.begin() + tests_num/10); // delete lower 10%
    time.erase(time.end() - tests_num/10, time.end());; // delete higher 10%

    double sum = std::accumulate(time.begin(), time.end(), 0.0);
    return sum / time.size();
}

int main(int argc, char** argv) {

    const int tests_num = 10;
    std::vector<int> test_threads{1,2,4,7,8,16,20,40}; // 1 - REQUIRED!

    std::ofstream fout("integrate_time.csv");
    fout << std::fixed << std::setprecision(3);

    fout << "threads, "
         << "time, "
         << "acceleration" << std::endl;

    double time1;

    for (int nthreads : test_threads) {
        double time;

        time = benchmark(tests_num, nthreads);
        time1 = (nthreads == 1) ? time : time1;
        fout << nthreads       << ", "
             << time           << ", "
             << time1/time << std::endl;
    }

    fout.close();

    return 0;
}
