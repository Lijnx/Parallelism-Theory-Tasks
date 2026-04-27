#include "server.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>


double random_double(std::mt19937& gen, double left, double right) {
    std::uniform_real_distribution<double> dist(left, right);
    return dist(gen);
}

int random_int(std::mt19937& gen, int left, int right) {
    std::uniform_int_distribution<int> dist(left, right);
    return dist(gen);
}


void client_sin(Server<double>& server, int n, const std::string& filename) {
    std::ofstream fout(filename);

    if (!fout) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    fout << std::setprecision(17);
    fout << "id,operation,x,result\n";

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < n; ++i) {
        double x = random_double(gen, -100.0, 100.0);

        std::size_t id = server.add_task([x] {
            return std::sin(x);
        });

        double result = server.request_result(id);

        fout << id << ",sin," << x << "," << result << "\n";
    }
}


void client_sqrt(Server<double>& server, int n, const std::string& filename) {
    std::ofstream fout(filename);

    if (!fout) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    fout << std::setprecision(17);
    fout << "id,operation,x,result\n";

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < n; ++i) {
        double x = random_double(gen, 0.0, 1000000.0);

        std::size_t id = server.add_task([x] {
            return std::sqrt(x);
        });

        double result = server.request_result(id);

        fout << id << ",sqrt," << x << "," << result << "\n";
    }
}


void client_pow(Server<double>& server, int n, const std::string& filename) {
    std::ofstream fout(filename);

    if (!fout) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    fout << std::setprecision(17);
    fout << "id,operation,x,power,result\n";

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < n; ++i) {
        double x = random_double(gen, 0.1, 100.0);
        int power = random_int(gen, 1, 8);

        std::size_t id = server.add_task([x, power] {
            return std::pow(x, power);
        });

        double result = server.request_result(id);

        fout << id << ",pow," << x << "," << power << "," << result << "\n";
    }
}


int main() {
    try {
        const int n = 10000;

        Server<double> server;
        server.start();

        std::thread client1(client_sin, std::ref(server), n, "sin_results.csv");
        std::thread client2(client_sqrt, std::ref(server), n, "sqrt_results.csv");
        std::thread client3(client_pow, std::ref(server), n, "pow_results.csv");

        client1.join();
        client2.join();
        client3.join();

        server.stop();

        std::cout << "Computation finished\n";
        std::cout << "Completed tasks: "
                  << server.completed_tasks()
                  << "\n";
        std::cout << "Server lifetime time: "
                  << server.server_lifetime_seconds()
                  << " seconds\n";
        std::cout << "Server task execution time: "
                  << server.total_task_time_seconds()
                  << " seconds\n";
        std::cout << "Generated files:\n";
        std::cout << "  sin_results.csv\n";
        std::cout << "  sqrt_results.csv\n";
        std::cout << "  pow_results.csv\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
