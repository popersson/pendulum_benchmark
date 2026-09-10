#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <mdspan>
#include <numeric>
#include <vector>

#include "static_vector.hpp"
#include "static_vector_ref.hpp"

constexpr std::size_t dimension = 4;

using State = static_vector<double, dimension>;
using StateRef = static_vector_ref<double, dimension>;
using Matrix = std::mdspan<double,
                           std::extents<std::size_t, dimension, std::dynamic_extent>,
                           std::layout_left>;

// layout_left keeps each column contiguous, so a column is just a reference to
// the dimension elements starting at its first one.  Assign to it to write the
// column, convert it to a State to read the column.
StateRef column(Matrix y, std::size_t n)
{
    return StateRef(&y[0, n]);
}

void runge5(const auto& f, const State& y0, double h, Matrix y)
{
    const std::size_t N = y.extent(1) - 1;

    column(y, 0) = y0;
    for (std::size_t n = 0; n < N; ++n) {
        const State yn = column(y, n);
        const State k1 = f(yn);
        const State k2 = f(yn + h * k1 / 5.0);
        const State k3 = f(yn + h * 2.0 * k2 / 5.0);
        const State k4 = f(yn + h * 9.0 * k1 / 4.0 - h * 5.0 * k2
                           + h * 15.0 * k3 / 4.0);
        const State k5 = f(yn - h * 63.0 * k1 / 100.0 + h * 9.0 * k2 / 5.0
                           - h * 13.0 * k3 / 20.0 + h * 2.0 * k4 / 25.0);
        const State k6 = f(yn - h * 6.0 * k1 / 25.0 + h * 4.0 * k2 / 5.0
                           + h * 2.0 * k3 / 15.0 + h * 8.0 * k4 / 75.0);
        column(y, n + 1) = yn + h * (17.0 * k1 + 100.0 * k3 + 2.0 * k4
                                     - 50.0 * k5 + 75.0 * k6) / 144.0;
    }
}

State fpend(const State& y)
{
    const auto [θ1, θ2, ω1, ω2] = y;

    // Each sin/cos pair of the same angle costs one sincos call; every other
    // trigonometric value the equations need follows by identity.
    const double s1 = std::sin(θ1), c1 = std::cos(θ1);
    const double s2 = std::sin(θ2), c2 = std::cos(θ2);
    const double sΔ = s1 * c2 - c1 * s2;              // sin(θ1 - θ2)
    const double cΔ = c1 * c2 + s1 * s2;              // cos(θ1 - θ2)
    const double s12 = sΔ * c2 - cΔ * s2;             // sin(θ1 - 2θ2)
    const double denominator = 2.0 + 2.0 * sΔ * sΔ;   // 3 - cos(2θ1 - 2θ2)

    const double θ1dot = ω1;
    const double θ2dot = ω2;
    const double ω1dot = (-3.0 * s1 - s12
                          - 2.0 * sΔ * (ω2 * ω2 + ω1 * ω1 * cΔ))
                         / denominator;
    const double ω2dot = 2.0 * sΔ * (2.0 * ω1 * ω1 + 2.0 * c1 + ω2 * ω2 * cΔ)
                         / denominator;

    return {θ1dot, θ2dot, ω1dot, ω2dot};
}

volatile double benchmark_sink;

int main()
{
    const State y0 = {2.0, 2.0, 0.0, -1.0};
    const double h = 0.2;
    const double T = 10000.0;
    const auto N = static_cast<std::size_t>(std::lround(T / h));

    for (int iter = 0; iter < 10; ++iter) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<double> storage(dimension * (N + 1));
        const Matrix y(storage.data(), N + 1);
        runge5(fpend, y0, h, y);
        const auto finish = std::chrono::steady_clock::now();

        // Keep every element observable without including the traversal in the timing.
        benchmark_sink = std::accumulate(storage.begin(), storage.end(), 0.0);

        const std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Time " << std::fixed << std::setprecision(6)
                  << elapsed.count() << " seconds.\n";
    }
}
