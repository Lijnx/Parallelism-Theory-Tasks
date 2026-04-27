#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ',')) {
        parts.push_back(item);
    }

    return parts;
}


TEST(ResultFiles, SinResultsAreCorrect) {
    std::ifstream fin("sin_results.csv");
    ASSERT_TRUE(fin.is_open()) << "Cannot open sin_results.csv";

    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(fin, line)));

    EXPECT_EQ(line, "id,operation,x,result");

    while (std::getline(fin, line)) {
        auto parts = split_csv_line(line);

        ASSERT_EQ(parts.size(), 4u);

        EXPECT_EQ(parts[1], "sin");

        double x = std::stod(parts[2]);
        double actual = std::stod(parts[3]);
        double expected = std::sin(x);

        EXPECT_NEAR(actual, expected, 1e-9)
            << "x = " << x
            << ", expected = " << expected
            << ", actual = " << actual;
    }
}


TEST(ResultFiles, SqrtResultsAreCorrect) {
    std::ifstream fin("sqrt_results.csv");
    ASSERT_TRUE(fin.is_open()) << "Cannot open sqrt_results.csv";

    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(fin, line)));

    EXPECT_EQ(line, "id,operation,x,result");

    while (std::getline(fin, line)) {
        auto parts = split_csv_line(line);

        ASSERT_EQ(parts.size(), 4u);

        EXPECT_EQ(parts[1], "sqrt");

        double x = std::stod(parts[2]);
        double actual = std::stod(parts[3]);
        double expected = std::sqrt(x);

        EXPECT_GE(x, 0.0);

        EXPECT_NEAR(actual, expected, 1e-9)
            << "x = " << x
            << ", expected = " << expected
            << ", actual = " << actual;

    }
}


TEST(ResultFiles, PowResultsAreCorrect) {
    std::ifstream fin("pow_results.csv");
    ASSERT_TRUE(fin.is_open()) << "Cannot open pow_results.csv";

    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(fin, line)));

    EXPECT_EQ(line, "id,operation,x,power,result");

    while (std::getline(fin, line)) {
        auto parts = split_csv_line(line);

        ASSERT_EQ(parts.size(), 5u);

        EXPECT_EQ(parts[1], "pow");

        double x = std::stod(parts[2]);
        int power = std::stoi(parts[3]);
        double actual = std::stod(parts[4]);
        double expected = std::pow(x, power);

        EXPECT_GE(x, 0.1);
        EXPECT_GE(power, 1);

        EXPECT_NEAR(actual, expected, 1e-9)
            << "x = " << x
            << ", power = " << power
            << ", expected = " << expected
            << ", actual = " << actual;

    }
}
