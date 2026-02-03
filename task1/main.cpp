#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <vector>

template<typename T>
T sinsum(std::vector<T> &vec) {
    T sum = 0;
    for (size_t i = 0; i < vec.size(); i++) {
        T dot = i * 2 * M_PI / vec.size();
        vec[i] = std::sin(dot);
        sum += vec[i];
    }
    return sum;
}

#ifdef USE_DOUBLE
using arr_t = double;
#else
using arr_t = float;
#endif

int main(int argc, char** argv) {

    size_t size = 10000000;
    std::vector<arr_t>vec(size);
    arr_t sum = sinsum(vec);
    std::cout << sum << std::endl;

    return 0;
}