// static_vector_ref.hpp -- a reference to a fixed-size vector we do not own.
//
// This is the non-owning half of the pair, in the same way that std::span is the
// non-owning half of std::array.  It holds nothing but a pointer to N contiguous
// elements: assigning to it writes the whole vector through to those elements,
// and converting it back yields a static_vector, so a column of a larger array
// is read and written with plain assignment instead of a hand-rolled loop.
//
// Instantiate with a const element type (static_vector_ref<const double, 4>) for
// a read-only reference; the assigning operations then do not exist.

#pragma once

#include <cstddef>
#include <type_traits>

#include "static_vector.hpp"

template <typename T, std::size_t N>
class static_vector_ref {
public:
    using element_type = T;
    using value_type = static_vector<std::remove_const_t<T>, N>;
    using size_type = std::size_t;
    using iterator = T*;

    static constexpr bool is_mutable = !std::is_const_v<T>;

    constexpr explicit static_vector_ref(T* elements) : elements(elements) {}
    constexpr explicit static_vector_ref(value_type& x) : elements(x.data()) {}
    constexpr explicit static_vector_ref(const value_type& x)
        requires (!is_mutable) : elements(x.data()) {}

    // Reading: materialize the referenced elements as a value.
    constexpr operator value_type() const
    {
        value_type x;
        for (size_type i = 0; i < N; ++i)
            x[i] = elements[i];
        return x;
    }

    constexpr value_type value() const { return *this; }

    // Writing: assign the vector as a whole, which is the point of the type.
    constexpr static_vector_ref& operator=(const value_type& x)
        requires is_mutable
    {
        for (size_type i = 0; i < N; ++i)
            elements[i] = x[i];
        return *this;
    }

    // Deep, like the assignment above: a reference does not rebind on assignment.
    constexpr static_vector_ref& operator=(const static_vector_ref& x)
        requires is_mutable
    {
        return *this = x.value();
    }

    constexpr static_vector_ref& operator+=(const value_type& x) requires is_mutable
    {
        for (size_type i = 0; i < N; ++i)
            elements[i] += x[i];
        return *this;
    }

    constexpr static_vector_ref& operator-=(const value_type& x) requires is_mutable
    {
        for (size_type i = 0; i < N; ++i)
            elements[i] -= x[i];
        return *this;
    }

    constexpr static_vector_ref& operator*=(const std::remove_const_t<T>& a)
        requires is_mutable
    {
        for (T& xi : *this)
            xi *= a;
        return *this;
    }

    constexpr static_vector_ref& operator/=(const std::remove_const_t<T>& a)
        requires is_mutable
    {
        for (T& xi : *this)
            xi /= a;
        return *this;
    }

    // Element access and iteration, so the reference is useful on its own.  The
    // constness here is that of the referenced elements, not of the reference.
    constexpr T& operator[](size_type i) const { return elements[i]; }
    constexpr T& front() const { return elements[0]; }
    constexpr T& back() const { return elements[N - 1]; }
    constexpr T* data() const { return elements; }
    constexpr T* begin() const { return elements; }
    constexpr T* end() const { return elements + N; }

    static constexpr size_type size() { return N; }
    static constexpr bool empty() { return N == 0; }

private:
    T* elements;
};

template <typename T, std::size_t N>
static_vector_ref(static_vector<T, N>&) -> static_vector_ref<T, N>;
