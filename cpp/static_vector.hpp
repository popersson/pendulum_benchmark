// static_vector.hpp -- a lean owning fixed-size numeric vector.
//
// C++23 has std::mdspan for *viewing* memory it does not own, but no owning
// counterpart (std::mdarray is a C++26 proposal).  static_vector fills that gap
// for rank one: it is an aggregate, so an object is exactly N elements with no
// indirection and no allocation, it initializes from a braced list, and it comes
// with the element-wise arithmetic that lets numerical code read like the
// mathematics it implements (compare StaticArrays.SVector in Julia).

#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <ostream>
#include <tuple>

template <typename T, std::size_t N>
struct static_vector {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    T elements[N];  // public, which is what keeps the type an aggregate

    // Element access.
    constexpr reference operator[](size_type i) { return elements[i]; }
    constexpr const_reference operator[](size_type i) const { return elements[i]; }
    constexpr reference front() { return elements[0]; }
    constexpr const_reference front() const { return elements[0]; }
    constexpr reference back() { return elements[N - 1]; }
    constexpr const_reference back() const { return elements[N - 1]; }
    constexpr T* data() { return elements; }
    constexpr const T* data() const { return elements; }

    // Size, known at compile time.
    static constexpr size_type size() { return N; }
    static constexpr bool empty() { return N == 0; }

    // Iterators, so the type works with the standard algorithms and ranges.
    constexpr iterator begin() { return elements; }
    constexpr iterator end() { return elements + N; }
    constexpr const_iterator begin() const { return elements; }
    constexpr const_iterator end() const { return elements + N; }
    constexpr const_iterator cbegin() const { return elements; }
    constexpr const_iterator cend() const { return elements + N; }

    // Whole-vector construction and assignment.
    static constexpr static_vector filled(const T& value)
    {
        static_vector v{};
        v.fill(value);
        return v;
    }

    static constexpr static_vector zeros() { return filled(T{}); }

    constexpr void fill(const T& value)
    {
        for (T& xi : elements)
            xi = value;
    }

    constexpr static_vector& operator+=(const static_vector& y)
    {
        for (size_type i = 0; i < N; ++i)
            elements[i] += y[i];
        return *this;
    }

    constexpr static_vector& operator-=(const static_vector& y)
    {
        for (size_type i = 0; i < N; ++i)
            elements[i] -= y[i];
        return *this;
    }

    constexpr static_vector& operator*=(const T& a)
    {
        for (T& xi : elements)
            xi *= a;
        return *this;
    }

    constexpr static_vector& operator/=(const T& a)
    {
        for (T& xi : elements)
            xi /= a;
        return *this;
    }
};

template <typename T, typename... U>
static_vector(T, U...) -> static_vector<T, 1 + sizeof...(U)>;

// A scalar is anything that converts to the element type, so that both
// 2 * x and 2.0 * x compile for a vector of doubles.
template <typename S, typename T>
concept scalar_for = std::convertible_to<const S&, T>;

template <typename T, std::size_t N>
constexpr static_vector<T, N> operator+(static_vector<T, N> x,
                                        const static_vector<T, N>& y)
{
    x += y;
    return x;
}

template <typename T, std::size_t N>
constexpr static_vector<T, N> operator-(static_vector<T, N> x,
                                        const static_vector<T, N>& y)
{
    x -= y;
    return x;
}

template <typename T, std::size_t N>
constexpr static_vector<T, N> operator+(static_vector<T, N> x)
{
    return x;
}

template <typename T, std::size_t N>
constexpr static_vector<T, N> operator-(static_vector<T, N> x)
{
    for (T& xi : x)
        xi = -xi;
    return x;
}

template <typename T, std::size_t N, scalar_for<T> S>
constexpr static_vector<T, N> operator*(const S& a, static_vector<T, N> x)
{
    x *= a;
    return x;
}

template <typename T, std::size_t N, scalar_for<T> S>
constexpr static_vector<T, N> operator*(static_vector<T, N> x, const S& a)
{
    x *= a;
    return x;
}

template <typename T, std::size_t N, scalar_for<T> S>
constexpr static_vector<T, N> operator/(static_vector<T, N> x, const S& a)
{
    x /= a;
    return x;
}

template <typename T, std::size_t N>
constexpr bool operator==(const static_vector<T, N>& x,
                          const static_vector<T, N>& y)
{
    for (std::size_t i = 0; i < N; ++i)
        if (!(x[i] == y[i]))
            return false;
    return true;
}

// Reductions.
template <typename T, std::size_t N>
constexpr T sum(const static_vector<T, N>& x)
{
    T total{};
    for (const T& xi : x)
        total += xi;
    return total;
}

template <typename T, std::size_t N>
constexpr T dot(const static_vector<T, N>& x, const static_vector<T, N>& y)
{
    T total{};
    for (std::size_t i = 0; i < N; ++i)
        total += x[i] * y[i];
    return total;
}

template <typename T, std::size_t N>
constexpr T norm(const static_vector<T, N>& x)
{
    return std::sqrt(dot(x, x));
}

// The tuple protocol, which is what makes structured bindings work:
//     const auto [x, y, z] = v;
template <std::size_t I, typename T, std::size_t N>
constexpr T& get(static_vector<T, N>& x)
{
    static_assert(I < N, "static_vector index out of range");
    return x[I];
}

template <std::size_t I, typename T, std::size_t N>
constexpr const T& get(const static_vector<T, N>& x)
{
    static_assert(I < N, "static_vector index out of range");
    return x[I];
}

template <std::size_t I, typename T, std::size_t N>
constexpr T&& get(static_vector<T, N>&& x)
{
    static_assert(I < N, "static_vector index out of range");
    return std::move(x[I]);
}

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& stream, const static_vector<T, N>& x)
{
    stream << '[';
    for (std::size_t i = 0; i < N; ++i)
        stream << (i == 0 ? "" : ", ") << x[i];
    return stream << ']';
}

template <typename T, std::size_t N>
struct std::tuple_size<static_vector<T, N>>
    : std::integral_constant<std::size_t, N> {};

template <std::size_t I, typename T, std::size_t N>
struct std::tuple_element<I, static_vector<T, N>> {
    using type = T;
};
