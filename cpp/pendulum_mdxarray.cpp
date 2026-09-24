#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "md.h"

constexpr md::index_t dimension = 4;

using State = md::sarray<double, dimension>;
using Matrix = md::dview<2>;   // md::darray<2> binds to this: array derives from view

// layout_left is Fortran order: the FIRST index varies fastest, so one state --
// a column -- is four consecutive doubles.  Fixing the last index therefore
// leaves a gap-free block, and that is exactly what page(n) hands back, for
// free.  Assigning to a slice copies elements; only a named view would rebind.
void runge5(const auto& f, const State& y0, double h, Matrix y)
{
    const md::index_t N = y.n(1) - 1;

    y.page(0) = y0;
    for (md::index_t n = 0; n < N; ++n) {
        const State yn(y.page(n));
        const State k1 = f(yn);
        const State k2 = f(yn + h * k1 / 5.0);
        const State k3 = f(yn + h * 2.0 * k2 / 5.0);
        const State k4 = f(yn + h * 9.0 * k1 / 4.0 - h * 5.0 * k2
                           + h * 15.0 * k3 / 4.0);
        const State k5 = f(yn - h * 63.0 * k1 / 100.0 + h * 9.0 * k2 / 5.0
                           - h * 13.0 * k3 / 20.0 + h * 2.0 * k4 / 25.0);
        const State k6 = f(yn - h * 6.0 * k1 / 25.0 + h * 4.0 * k2 / 5.0
                           + h * 2.0 * k3 / 15.0 + h * 8.0 * k4 / 75.0);
        y.page(n + 1) = yn + h * (17.0 * k1 + 100.0 * k3 + 2.0 * k4
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
    const auto N = static_cast<md::index_t>(std::lround(T / h));

    for (int iter = 0; iter < 10; ++iter) {
        const auto start = std::chrono::steady_clock::now();
        md::darray<2> y(dimension, N + 1);
        // Pass a lambda rather than fpend itself: a function reference binds as a
        // pointer, so clang cannot inline the six calls per step.  Worth ~2%.
        runge5([](const State& state) { return fpend(state); }, y0, h, y);
        const auto finish = std::chrono::steady_clock::now();

        // Keep every element observable without including the traversal in the timing.
        benchmark_sink = md::sum(y);

        const std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Time " << std::fixed << std::setprecision(6)
                  << elapsed.count() << " seconds.\n";
    }
}
