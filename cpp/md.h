// md.h -- mdxarray: multidimensional arrays and views built on std::mdspan.
//
// Four types, along two axes:
//
//                       non-owning              owning
//     dynamic extents   md::view<T,R>           md::array<T,R>
//     static extents    md::sview<T,E...>       md::sarray<T,E...>
//
// All of them use layout_left (Fortran order) and a signed 32-bit index.
// Index with A[i,j,k]; A(i,j,k) is accepted too and means the same thing.
//
// SPDX-License-Identifier: MIT
#pragma once

// The rest of this header is guarded so that a missing or too-old <mdspan>
// produces exactly one diagnostic instead of that one followed by several
// hundred lines of fallout.
#if !__has_include(<mdspan>)

#  error "mdxarray requires <mdspan> (C++23). libstdc++ does not ship it yet; build with clang and libc++:  clang++ -std=c++23 -stdlib=libc++"

#else
#include <mdspan>
#if !defined(__cpp_lib_mdspan)

#  error "mdxarray requires <mdspan> (C++23). Compile with -std=c++23, and -stdlib=libc++ on Linux."

#else

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>

namespace md {

// Signed, 32-bit: it matches the classic F77 LAPACK ABI, and it keeps a rank-2
// view at 16 bytes, which is the x86-64 threshold for passing in registers.
// Change this one line to std::int64_t for arrays past 2^31 elements or for an
// ILP64 BLAS; nothing else in the header depends on the width.
using index_t = int;

template <class T, std::size_t R> using dext = std::dextents<index_t, R>;
template <class T, index_t... E>  using sext = std::extents<index_t, E...>;

// Optional index and shape checking, off by default.
//
// Define MD_BOUNDS_CHECK to enable it. These checks guard element access and
// the elementwise algorithms, which are called from inner loops, so they are
// opt-in rather than tied to NDEBUG: md::assign inside a per-node loop runs
// once per node, and an unconditional check there measured 2x on a nodal flux
// kernel. Operations called once per matrix (reshape, the LAPACK wrappers) use
// a plain assert instead, where the cost is irrelevant.
#ifdef MD_BOUNDS_CHECK
#  define MD_ASSERT(cond, msg) assert((cond) && (msg))
#else
#  define MD_ASSERT(cond, msg) ((void)0)
#endif

// ===========================================================================
// basic_view -- one view class for every combination of extents and layout.
// Dynamic vs. static is nothing but a different Extents type.
//
// A view is a handle: copying it is copying a pointer and a shape, never data.
// Read-only is expressed in the ELEMENT TYPE (view<const T,R>), not by const on
// the view itself, exactly as for std::mdspan and for a plain T* const.
// ===========================================================================
template <class D, class S> constexpr void assign(D&& d, const S& s);

template <class T, class Ext, class Layout = std::layout_left>
struct basic_view : std::mdspan<T, Ext, Layout> {
  using base      = std::mdspan<T, Ext, Layout>;
  using element_t = T;
  using extents_t = Ext;
  using layout_t  = Layout;
  using base::base;

  // True when the elements form a gap-free block, which is what allows a flat
  // loop over data(). Always true for layout_left; false for a strided slice.
  static constexpr bool contiguous = base::mapping_type::is_always_exhaustive();
  static constexpr std::size_t rank_v = Ext::rank();

  constexpr basic_view() = default;
  constexpr basic_view(const base& b) : base(b) {}
  constexpr basic_view(const basic_view&) = default;

  // ---- assignment --------------------------------------------------------
  // Assignment is split by value category, because the two plausible meanings
  // apply to disjoint cases:
  //
  //     named lvalue   A = B;              rebind the handle, as mdspan does
  //     temporary      A.page(k) = expr;   write through to the elements
  //
  // A slice expression yields a temporary, and rebinding a temporary cannot do
  // anything observable -- before this split, `A.page(k) = B.page(j)` compiled,
  // copied nothing, and warned about nothing. So the rule is:
  //
  //     assigning to a NAME rebinds; assigning to a SLICE writes.
  //
  // Nothing that previously worked changes meaning, because the only case whose
  // behaviour changes is the one that previously did nothing.
  constexpr basic_view& operator=(const basic_view& o) & = default;

  template <class S>
    requires requires (const S& s) { s.data(); s.nelem(); }
  constexpr void operator=(const S& s) && {
    MD_ASSERT(nelem() == s.nelem(), "md: assignment shapes do not match");
    md::assign(*this, s);
  }

  constexpr void operator=(std::type_identity_t<T> x) && {
    T* p = data(); const index_t n = nelem();
    for (index_t i = 0; i < n; ++i) p[i] = x;
  }

  // T -> const T, never the reverse.
  template <class U>
    requires std::is_same_v<T, const U>
  constexpr basic_view(basic_view<U, Ext, Layout> b) : base(b.data_handle(), b.mapping()) {}

  // When every extent is static the shape already lives in the type, so a bare
  // pointer supplies everything that is missing. Non-explicit on purpose: it
  // makes f(&A[0,0,k]) a one-liner when f takes an sview<double,4,5>. The
  // caller is asserting that the elements are contiguous from that address.
  constexpr basic_view(T* p) requires (Ext::rank_dynamic() == 0) : base(p) {}

  // ---- element access ----------------------------------------------------
  // operator[] is the primary accessor and carries the bounds check; it hides
  // the inherited mdspan one, which is unchecked. operator() is a synonym.
  template <class... I>
    requires (sizeof...(I) == Ext::rank())
  constexpr T& operator[](I... i) const {
    MD_ASSERT(in_bounds(static_cast<index_t>(i)...), "md: index out of bounds");
    return base::operator[](static_cast<index_t>(i)...);
  }
  template <class... I>
    requires (sizeof...(I) == Ext::rank())
  constexpr T& operator()(I... i) const { return (*this)[i...]; }

  // ---- shape and storage -------------------------------------------------
  constexpr T* data() const { return this->data_handle(); }
  constexpr index_t n(std::size_t r) const { return static_cast<index_t>(this->extent(r)); }

  // Number of ELEMENTS (product of the extents). For a strided view this is
  // smaller than the memory span it covers -- see span_size().
  constexpr index_t nelem() const { return static_cast<index_t>(this->size()); }

  // Size of the memory region the mapping can reach. Equals nelem() when
  // contiguous; larger for a strided slice.
  constexpr index_t span_size() const
  { return static_cast<index_t>(this->mapping().required_span_size()); }

  constexpr base to_mdspan() const { return *this; }

  // A null view tests false, so `if (A && B)` reads naturally: && applies a
  // contextual conversion, so explicit is enough.
  constexpr explicit operator bool() const { return this->data_handle() != nullptr; }

  // Flat iteration is only meaningful for a gap-free layout. Constraining it
  // turns "treat a strided slice as contiguous" into a compile error.
  constexpr T* begin() const requires contiguous { return data(); }
  constexpr T* end()   const requires contiguous { return data() + nelem(); }

  // ---- slicing -----------------------------------------------------------
  // Two mirror-image slices, each returning a rank-(R-1) view.
  //
  //   page(k)  fixes the LAST index. Under layout_left this is the only slice
  //            that is always contiguous, so it comes back as a plain view and
  //            costs nothing. This is the &A[0,0,k] idiom.
  //   row(i)   fixes the FIRST index. Necessarily strided, so it comes back as
  //            a strided_view: no flat iteration, no fast arithmetic, but the
  //            layout-agnostic algorithms below all accept it.
  constexpr auto page(index_t k) const
    requires (Ext::rank() >= 2 && contiguous)
  {
    MD_ASSERT(k >= 0 && k < n(Ext::rank() - 1), "md: page index out of bounds");
    return page_impl(k, std::make_index_sequence<Ext::rank() - 1>{});
  }

  constexpr auto row(index_t i) const
    requires (Ext::rank() >= 2)
  {
    MD_ASSERT(i >= 0 && i < n(0), "md: row index out of bounds");
    return row_impl(i, std::make_index_sequence<Ext::rank() - 1>{});
  }

 private:
  template <class... I>
  constexpr bool in_bounds(I... i) const {
    index_t r = 0;
    return ((static_cast<index_t>(i) >= 0 &&
             static_cast<index_t>(i) < n(r++)) && ...);
  }
  template <std::size_t... K>
  constexpr auto page_impl(index_t k, std::index_sequence<K...>) const {
    constexpr std::size_t R = Ext::rank();
    using sub_t = basic_view<T, std::dextents<index_t, R - 1>, std::layout_left>;
    return sub_t(data() + k * static_cast<index_t>(this->stride(R - 1)),
                 static_cast<index_t>(this->extent(K))...);
  }
  template <std::size_t... K>
  constexpr auto row_impl(index_t i, std::index_sequence<K...>) const {
    constexpr std::size_t R = Ext::rank();
    using sub_ext = std::dextents<index_t, R - 1>;
    using sub_t   = basic_view<T, sub_ext, std::layout_stride>;
    sub_ext e(static_cast<index_t>(this->extent(K + 1))...);
    std::array<index_t, R - 1> st{static_cast<index_t>(this->stride(K + 1))...};
    return sub_t(data() + i * static_cast<index_t>(this->stride(0)),
                 std::layout_stride::mapping<sub_ext>(e, st));
  }
};

template <class T, std::size_t R> using view          = basic_view<T, dext<T, R>>;
template <class T, index_t... E>  using sview         = basic_view<T, sext<T, E...>>;
template <class T, std::size_t R> using strided_view  = basic_view<T, dext<T, R>,    std::layout_stride>;
template <class T, index_t... E>  using strided_sview = basic_view<T, sext<T, E...>, std::layout_stride>;

// ===========================================================================
// array -- owning, dynamic extents.
//
// It derives from its own view. That is load-bearing rather than decorative:
// template argument deduction runs before user-defined conversions, so a
// view<T,R> parameter would never accept an array<T,R> by conversion, but
// deduction does consider derived-to-base. The inheritance is what lets a
// function take a view and a caller pass an array, and what lets A += B work
// with arrays on either side.
//
// Move-only. array.h made `array B = A;` a silent non-owning view; here a copy
// must be spelled A.copy() and a transfer must be spelled std::move(A).
// ===========================================================================
template <class T, std::size_t R>
struct array : view<T, R> {
  using view_t    = view<T, R>;
  using cview_t   = view<const T, R>;
  using map_t     = typename view_t::base::mapping_type;
  using element_t = T;

  array() = default;

  // Rank comes from the number of arguments. Storage is left UNINITIALIZED.
  template <class... I>
    requires (sizeof...(I) == R) && (std::is_convertible_v<I, index_t> && ...)
  explicit array(I... n) {
    map_t m{dext<T, R>(static_cast<index_t>(n)...)};
    p_ = std::make_unique_for_overwrite<T[]>(
             static_cast<std::size_t>(m.required_span_size()));
    static_cast<view_t&>(*this) = view_t(p_.get(), m);
  }

  // Hand-written rather than defaulted: a defaulted move would copy the view
  // half and move the pointer out, leaving the moved-from object's view aimed
  // at freed memory while still testing true.
  array(array&& o) noexcept : view_t(o), p_(std::move(o.p_)) {
    static_cast<view_t&>(o) = view_t{};
  }
  array& operator=(array&& o) noexcept {
    if (this != &o) {
      static_cast<view_t&>(*this) = o;
      p_ = std::move(o.p_);
      static_cast<view_t&>(o) = view_t{};
    }
    return *this;
  }
  array(const array&)            = delete;
  array& operator=(const array&) = delete;

  array copy() const {
    array r;
    if (this->data()) {
      const auto n = static_cast<std::size_t>(this->span_size());
      r.p_ = std::make_unique_for_overwrite<T[]>(n);
      std::copy_n(this->data(), n, r.p_.get());
      static_cast<view_t&>(r) = view_t(r.p_.get(), this->mapping());
    }
    return r;
  }

  void swap(array& o) noexcept {
    view_t tmp = *this;
    static_cast<view_t&>(*this) = o;
    static_cast<view_t&>(o) = tmp;
    p_.swap(o.p_);
  }

  // Const propagates, as it does for std::mdarray: an owning object that is
  // const hands out const elements. These hide the base's accessors, which are
  // const members returning T& because a view is a handle.
  //
  // Note that this protects direct element access only. A free function taking
  // a view parameter binds to the mutable base subobject, so a read-only
  // INTERFACE should say so in the element type: take view<const T,R>.
  template <class... I> constexpr T& operator[](I... i)
  { return view_t::operator[](i...); }
  template <class... I> constexpr const T& operator[](I... i) const
  { return view_t::operator[](i...); }
  template <class... I> constexpr T& operator()(I... i)
  { return view_t::operator[](i...); }
  template <class... I> constexpr const T& operator()(I... i) const
  { return view_t::operator[](i...); }

  constexpr T*       data()       { return p_.get(); }
  constexpr const T* data() const { return p_.get(); }
  constexpr T*       begin()       { return p_.get(); }
  constexpr T*       end()         { return p_.get() + this->nelem(); }
  constexpr const T* begin() const { return p_.get(); }
  constexpr const T* end()   const { return p_.get() + this->nelem(); }

  constexpr view_t  view()  { return static_cast<view_t&>(*this); }
  constexpr cview_t cview() const { return cview_t(p_.get(), this->mapping()); }

  // ---- resizing ----------------------------------------------------------
  //   reshape(...)  same element count, data kept, never allocates
  //   resize(...)   always reallocates, contents undefined afterwards
  //   ensure(...)   reshape when the count matches, resize when it does not
  //
  // reshape/resize follow numpy and Eigen. ensure() has no counterpart
  // elsewhere; it is the one to call in a time-stepping loop where the size
  // usually does not change. (std::mdarray has no resizing at all.)
  template <class... I>
    requires (sizeof...(I) == R)
  void reshape(I... n) {
    map_t m{dext<T, R>(static_cast<index_t>(n)...)};
    assert(this->data() && m.required_span_size() == this->span_size() &&
           "md::array::reshape must preserve the element count; use resize or ensure");
    static_cast<view_t&>(*this) = view_t(p_.get(), m);
  }
  template <class... I>
    requires (sizeof...(I) == R)
  void resize(I... n) { *this = array(n...); }

  template <class... I>
    requires (sizeof...(I) == R)
  void ensure(I... n) {
    map_t m{dext<T, R>(static_cast<index_t>(n)...)};
    if (!this->data() || m.required_span_size() != this->span_size())
      *this = array(n...);
    else
      static_cast<view_t&>(*this) = view_t(p_.get(), m);
  }

 private:
  std::unique_ptr<T[]> p_{};
};

template <class T, std::size_t R>
void swap(array<T, R>& a, array<T, R>& b) noexcept { a.swap(b); }

// Element type explicit, rank deduced from the argument count. A deduction
// guide cannot leave a template parameter undeduced, so there is no way to
// write one guide that fixes the rank and leaves the element type open; the
// per-type aliases at the bottom of this header carry their own guides.
template <class T, class... I>
auto make(I... n) { return array<T, sizeof...(I)>(static_cast<index_t>(n)...); }

// ===========================================================================
// sarray -- owning, static extents.
//
// Unlike array this does NOT derive from its view: it would then hold a pointer
// into its own storage, and any copy or move would leave that pointer aimed at
// the original object. It holds storage only, and converts to sview on demand.
// ===========================================================================
template <class T, index_t... E>
struct sarray {
  using element_t = T;
  using extents_t = sext<T, E...>;
  using map_t     = std::layout_left::mapping<extents_t>;

  static constexpr index_t nelem_v   = (E * ...);
  static constexpr bool    contiguous = true;      // storage is a plain T[]
  static constexpr std::size_t rank() { return sizeof...(E); }

  sarray() = default;
  explicit constexpr sarray(T x) { for (auto& e : d_) e = x; }

  // Element-wise construction: sarray<double,4>{a, b, c, d}. The argument count
  // is checked at compile time and the constructor stays constexpr. Excluded at
  // nelem_v == 1, where it would tie with the fill constructor above and mean
  // the same thing anyway.
  template <class... U>
    requires (nelem_v > 1) && (sizeof...(U) == std::size_t(nelem_v)) &&
             (std::convertible_to<U, T> && ...)
  constexpr sarray(U... x) : d_{static_cast<T>(x)...} {}

  // A view whose shape matches at compile time compacts implicitly. Templated
  // on the view's element type on purpose: a parameter of sview<const T,E...>
  // would require sview<T,...> -> sview<const T,...> -> sarray, two
  // user-defined conversions, which the language will not perform.
  template <class U>
    requires std::is_same_v<std::remove_const_t<U>, T>
  constexpr sarray(sview<U, E...> v)
  { for (index_t i = 0; i < nelem_v; ++i) d_[i] = v.data()[i]; }

  // A dynamic-extent view compacts too, but its shape is only known at run
  // time, so this one is explicit: a size mismatch should not be something an
  // implicit conversion can hide.
  template <class V>
    requires requires (const V& v) { v.data(); v.nelem(); } &&
             std::is_same_v<std::remove_const_t<typename V::element_t>, T> &&
             (V::extents_t::rank_dynamic() > 0)
  explicit constexpr sarray(const V& v) {
    MD_ASSERT(v.nelem() == nelem_v, "md::sarray: source shape does not match");
    for (index_t i = 0; i < nelem_v; ++i) d_[i] = v.data()[i];
  }

  template <class... I> constexpr T& operator[](I... i) {
    MD_ASSERT(in_bounds(static_cast<index_t>(i)...), "md: index out of bounds");
    return d_[map_t{}(static_cast<index_t>(i)...)];
  }
  template <class... I> constexpr const T& operator[](I... i) const {
    MD_ASSERT(in_bounds(static_cast<index_t>(i)...), "md: index out of bounds");
    return d_[map_t{}(static_cast<index_t>(i)...)];
  }
  template <class... I> constexpr T&       operator()(I... i)       { return (*this)[i...]; }
  template <class... I> constexpr const T& operator()(I... i) const { return (*this)[i...]; }

  constexpr T*       data()        { return d_; }
  constexpr const T* data()  const { return d_; }
  constexpr T*       begin()       { return d_; }
  constexpr T*       end()         { return d_ + nelem_v; }
  constexpr const T* begin() const { return d_; }
  constexpr const T* end()   const { return d_ + nelem_v; }

  constexpr index_t nelem()     const { return nelem_v; }
  constexpr index_t span_size() const { return nelem_v; }
  constexpr index_t n(std::size_t r) const { return extents_t{}.extent(r); }
  static constexpr extents_t extents() { return extents_t{}; }

  constexpr operator sview<T, E...>()             { return sview<T, E...>(d_); }
  constexpr operator sview<const T, E...>() const { return sview<const T, E...>(d_); }

  constexpr sarray& operator+=(sview<const T, E...> b) { for (index_t i=0;i<nelem_v;++i) d_[i] += b.data()[i]; return *this; }
  constexpr sarray& operator-=(sview<const T, E...> b) { for (index_t i=0;i<nelem_v;++i) d_[i] -= b.data()[i]; return *this; }
  constexpr sarray& operator*=(sview<const T, E...> b) { for (index_t i=0;i<nelem_v;++i) d_[i] *= b.data()[i]; return *this; }
  constexpr sarray& operator/=(sview<const T, E...> b) { for (index_t i=0;i<nelem_v;++i) d_[i] /= b.data()[i]; return *this; }
  constexpr sarray& operator+=(T s) { for (auto& e : d_) e += s; return *this; }
  constexpr sarray& operator-=(T s) { for (auto& e : d_) e -= s; return *this; }
  constexpr sarray& operator*=(T s) { for (auto& e : d_) e *= s; return *this; }
  constexpr sarray& operator/=(T s) { for (auto& e : d_) e /= s; return *this; }

  // Value-returning arithmetic exists only here, where the result lives on the
  // stack. The dynamic types deliberately have no operator+, which makes a
  // hidden heap allocation impossible to write by accident and removes any
  // reason for expression templates.
  friend constexpr sarray operator+(sarray a, sview<const T, E...> b) { return a += b; }
  friend constexpr sarray operator-(sarray a, sview<const T, E...> b) { return a -= b; }
  friend constexpr sarray operator*(sarray a, T s)                    { return a *= s; }
  friend constexpr sarray operator*(T s, sarray a)                    { return a *= s; }
  friend constexpr sarray operator/(sarray a, T s)                    { return a /= s; }
  friend constexpr sarray operator-(sarray a) { for (auto& e : a.d_) e = -e; return a; }

  // Structured bindings, for rank 1: `auto [a, b, c, d] = state;`. Deliberately
  // not offered at higher rank, where the unpacking order would have to be
  // memorised rather than read.
  template <std::size_t I> requires (sizeof...(E) == 1)
  constexpr T& get() & { static_assert(I < std::size_t(nelem_v)); return d_[I]; }
  template <std::size_t I> requires (sizeof...(E) == 1)
  constexpr const T& get() const& { static_assert(I < std::size_t(nelem_v)); return d_[I]; }
  template <std::size_t I> requires (sizeof...(E) == 1)
  constexpr T&& get() && { static_assert(I < std::size_t(nelem_v)); return std::move(d_[I]); }

  T d_[nelem_v];

 private:
  template <class... I>
  static constexpr bool in_bounds(I... i) {
    index_t r = 0;
    return ((static_cast<index_t>(i) >= 0 &&
             static_cast<index_t>(i) < extents_t{}.extent(r++)) && ...);
  }
};

// ===========================================================================
// Traits and generic iteration
// ===========================================================================
namespace detail {

// `||` does not short-circuit instantiation inside a constexpr initializer, so
// the fallback needs if constexpr rather than one boolean expression.
template <class A> constexpr bool is_contiguous() {
  if constexpr (requires { A::contiguous; }) return A::contiguous;
  else return false;
}

template <class Ext, class F, std::size_t... K>
constexpr void each_index(const Ext& e, F&& f, std::index_sequence<K...>) {
  std::array<index_t, Ext::rank()> ix{};
  index_t total = 1;
  ((total *= static_cast<index_t>(e.extent(K))), ...);
  for (index_t lin = 0; lin < total; ++lin) {
    f(ix[K]...);
    for (std::size_t d = 0; d < Ext::rank(); ++d) {   // first index fastest
      if (++ix[d] < static_cast<index_t>(e.extent(d))) break;
      ix[d] = 0;
    }
  }
}

// Four independent accumulators. A single-accumulator loop carries a floating
// point dependency, and since FP addition is not associative the compiler may
// not vectorize it without -ffast-math: measured 17 GB/s naive against 60 GB/s
// unrolled. The summation order is fixed, so results stay reproducible.
template <class T> constexpr T sum4(const T* x, index_t n) {
  T s0{}, s1{}, s2{}, s3{}; index_t i = 0;
  for (; i + 3 < n; i += 4) { s0 += x[i]; s1 += x[i+1]; s2 += x[i+2]; s3 += x[i+3]; }
  T s = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) s += x[i];
  return s;
}
template <class T> constexpr T dot4(const T* x, const T* y, index_t n) {
  T s0{}, s1{}, s2{}, s3{}; index_t i = 0;
  for (; i + 3 < n; i += 4)
    { s0 += x[i]*y[i]; s1 += x[i+1]*y[i+1]; s2 += x[i+2]*y[i+2]; s3 += x[i+3]*y[i+3]; }
  T s = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) s += x[i]*y[i];
  return s;
}

}  // namespace detail

template <class A> inline constexpr bool is_contiguous_v = detail::is_contiguous<A>();

namespace detail {
template <class X, std::size_t... K>
constexpr index_t static_prod(std::index_sequence<K...>) {
  index_t n = 1;
  ((n *= static_cast<index_t>(X::static_extent(K))), ...);
  return n;
}
}  // namespace detail

// The element count as a compile-time constant when the type carries one, else
// -1. assign() picks it up from whichever operand has static extents, so
// copying a fixed-size sarray into a dynamic-extent slice still emits a
// fixed-length loop rather than one whose bound has to be loaded.
template <class A> constexpr index_t static_nelem() {
  using P = std::remove_cvref_t<A>;
  if constexpr (requires { P::nelem_v; }) return P::nelem_v;
  else if constexpr (requires { typename P::extents_t; }) {
    using X = typename P::extents_t;
    if constexpr (X::rank_dynamic() == 0)
      return detail::static_prod<X>(std::make_index_sequence<X::rank()>{});
    else return -1;
  } else return -1;
}
template <class A> using elem_t = std::remove_const_t<typename std::remove_cvref_t<A>::element_t>;

// Walk every multi-index of `e`, first index fastest (layout_left order).
template <class Ext, class F>
constexpr void for_each_index(const Ext& e, F&& f)
{ detail::each_index(e, static_cast<F&&>(f), std::make_index_sequence<Ext::rank()>{}); }

// Visit every element. Takes the flat path when the layout allows it.
template <class A, class F>
constexpr void for_each(A&& a, F&& f) {
  using Ap = std::remove_cvref_t<A>;
  if constexpr (is_contiguous_v<Ap>) {
    auto* p = a.data();
    const index_t n = a.nelem();
    for (index_t i = 0; i < n; ++i) f(p[i]);
  } else {
    for_each_index(a.extents(), [&](auto... i) { f(a[i...]); });
  }
}

// ===========================================================================
// Elementwise compound assignment on views.
//
// Two details, both load-bearing: the right operand is std::type_identity_t,
// i.e. non-deduced, so that an array is allowed to convert to a view there;
// and the left operand is deduced, which reaches an array through its view
// base. Drop either and the mixed array/view combinations stop compiling.
//
// Constrained on `contiguous`. For a strided slice use md::assign or md::apply.
// ===========================================================================
#define MD_OP(op)                                                              \
template <class T, class Ext, class L>                                         \
  requires basic_view<T, Ext, L>::contiguous                                   \
constexpr const basic_view<T,Ext,L>& operator op(                              \
    const basic_view<T,Ext,L>& a,                                              \
    std::type_identity_t<basic_view<const T,Ext,L>> b) {                       \
  T* x = a.data(); const T* y = b.data(); const index_t n = a.nelem();         \
  MD_ASSERT(n == b.nelem(), "md: operand shapes do not match");                \
  for (index_t i = 0; i < n; ++i) x[i] op y[i];                                \
  return a; }                                                                  \
template <class T, class Ext, class L>                                         \
  requires basic_view<T, Ext, L>::contiguous                                   \
constexpr const basic_view<T,Ext,L>& operator op(const basic_view<T,Ext,L>& a,  \
                                                 std::type_identity_t<T> s) {  \
  T* x = a.data(); const index_t n = a.nelem();                                \
  for (index_t i = 0; i < n; ++i) x[i] op s;                                   \
  return a; }
MD_OP(+=) MD_OP(-=) MD_OP(*=) MD_OP(/=)
#undef MD_OP

// ===========================================================================
// Algorithms. All of these accept a strided slice as well as a contiguous one.
//
// Free functions rather than members, following the direction of WG21 P3242
// ("Copy and fill for mdspan"), so md::fill has a path to becoming std::fill.
// ===========================================================================
template <class A>
constexpr A&& fill(A&& a, elem_t<A> s)
{ for_each(a, [&](auto& e) { e = s; }); return static_cast<A&&>(a); }

template <class F, class A>
constexpr A&& apply(F f, A&& a)
{ for_each(a, [&](auto& e) { e = f(e); }); return static_cast<A&&>(a); }

// dst <- src, elementwise. Either side may be strided.
template <class D, class S>
constexpr void assign(D&& d, const S& s) {
  using Dp = std::remove_cvref_t<D>;
  if constexpr (is_contiguous_v<Dp> && is_contiguous_v<S>) {
    auto* x = d.data(); const auto* y = s.data();
    MD_ASSERT(d.nelem() == s.nelem(), "md::assign: shapes do not match");
    constexpr index_t sd = static_nelem<Dp>(), ss = static_nelem<S>();
    constexpr index_t sn = sd > 0 ? sd : ss;
    if constexpr (sn > 0) {
      for (index_t i = 0; i < sn; ++i) x[i] = y[i];
    } else {
      const index_t n = d.nelem();
      for (index_t i = 0; i < n; ++i) x[i] = y[i];
    }
  } else {
    MD_ASSERT(d.nelem() == s.nelem(), "md::assign: shapes do not match");
    for_each_index(d.extents(), [&](auto... i) { d[i...] = s[i...]; });
  }
}

// Copy a (possibly strided) view into a compact static array. The shape is
// given explicitly because a dynamic view has no compile-time extents to offer.
template <index_t... E, class V>
constexpr auto compact(const V& v) {
  sarray<elem_t<V>, E...> r;
  assign(r, v);
  return r;
}

// ---- reductions -----------------------------------------------------------
template <class A> constexpr elem_t<A> sum(const A& a) {
  using T = elem_t<A>;
  if constexpr (is_contiguous_v<A>) return detail::sum4(a.data(), a.nelem());
  else { T s{}; for_each_index(a.extents(), [&](auto... i) { s += a[i...]; }); return s; }
}

template <class A, class B> constexpr elem_t<A> dot(const A& a, const B& b) {
  using T = elem_t<A>;
  MD_ASSERT(a.nelem() == b.nelem(), "md::dot: shapes do not match");
  if constexpr (is_contiguous_v<A> && is_contiguous_v<B>)
    return detail::dot4(a.data(), b.data(), a.nelem());
  else { T s{}; for_each_index(a.extents(), [&](auto... i) { s += a[i...] * b[i...]; }); return s; }
}

template <class A> constexpr double norm(const A& a) { return std::sqrt(double(dot(a, a))); }

template <class A> constexpr double infnorm(const A& a) {
  double m = 0;
  for_each(a, [&](const auto& e) { m = std::max(m, std::abs(double(e))); });
  return m;
}
template <class A> constexpr elem_t<A> maxval(const A& a) {
  using T = elem_t<A>;
  T m = std::numeric_limits<T>::lowest();
  for_each(a, [&](const auto& e) { if (e > m) m = e; });
  return m;
}
template <class A> constexpr elem_t<A> minval(const A& a) {
  using T = elem_t<A>;
  T m = (std::numeric_limits<T>::max)();
  for_each(a, [&](const auto& e) { if (e < m) m = e; });
  return m;
}
template <class A> constexpr bool anynan(const A& a) {
  bool found = false;
  for_each(a, [&](const auto& e) { if (std::isnan(double(e))) found = true; });
  return found;
}

// ---- broadcasting ---------------------------------------------------------
template <class F, class T, index_t... E>
constexpr auto map(F f, const sarray<T,E...>& a) {
  sarray<decltype(f(std::declval<T>())), E...> r;
  for (index_t i = 0; i < a.nelem_v; ++i) r.data()[i] = f(a.data()[i]);
  return r;
}
template <class F, class T, index_t... E>
constexpr auto map(F f, const sarray<T,E...>& a, const sarray<T,E...>& b) {
  sarray<decltype(f(std::declval<T>(), std::declval<T>())), E...> r;
  for (index_t i = 0; i < a.nelem_v; ++i) r.data()[i] = f(a.data()[i], b.data()[i]);
  return r;
}

// Named function objects for the standard math functions. md::map(std::sin, A)
// cannot work -- std::sin is an overload set, so the F template parameter has
// nothing to deduce from. These wrap the overload set in an object, giving both
// md::map(md::sin, A) and the shorter md::sin(A). Codegen is identical to a
// lambda or a cast function pointer; this is purely about what you type.
#define MD_FN(name)                                                            \
  inline constexpr struct name##_fn {                                          \
    template <class X> requires std::is_arithmetic_v<X>                        \
    constexpr auto operator()(X x) const { using std::name; return name(x); }  \
    template <class T, index_t... E>                                           \
    constexpr auto operator()(const sarray<T,E...>& a) const { return map(*this, a); } \
  } name{};
MD_FN(sin)  MD_FN(cos)  MD_FN(tan)   MD_FN(asin) MD_FN(acos) MD_FN(atan)
MD_FN(exp)  MD_FN(log)  MD_FN(log10) MD_FN(sqrt) MD_FN(cbrt)
MD_FN(sinh) MD_FN(cosh) MD_FN(tanh)  MD_FN(abs)  MD_FN(floor) MD_FN(ceil)
#undef MD_FN

// ---- debug printing -------------------------------------------------------
template <class A>
std::ostream& print(std::ostream& os, const A& a, const char* name = nullptr) {
  if (name) os << name << " = ";
  os << "[";
  for (std::size_t r = 0; r < std::remove_cvref_t<A>::rank(); ++r)
    os << (r ? "x" : "") << a.n(r);
  os << "]\n";
  for_each_index(a.extents(), [&](auto... i) {
    os << "  [";
    index_t k = 0;
    ((os << (k++ ? "," : "") << i), ...);
    os << "] " << a[i...] << "\n";
  });
  return os;
}

template <class T, class Ext, class L>
std::ostream& operator<<(std::ostream& os, const basic_view<T,Ext,L>& a) { return print(os, a); }
template <class T, index_t... E>
std::ostream& operator<<(std::ostream& os, const sarray<T,E...>& a) { return print(os, a); }

// ===========================================================================
// Short aliases.
//
// The owning ones are thin derived classes rather than alias templates, so each
// can carry its own deduction guide: that is what makes both `darray A(M,N)`
// and `farray A(M,N)` deduce their rank from the argument count. Views are
// plain alias templates -- they need no guide, since their element type comes
// from the pointer.
//
// These live in namespace md, so bring in what you want:  using md::darray;
// ===========================================================================
#define MD_ARRAY_ALIAS(name, T)                                                \
  template <std::size_t R> struct name : array<T, R> {                         \
    using array<T, R>::array;                                                  \
    name() = default;                                                          \
    name(array<T, R>&& a) noexcept : array<T, R>(std::move(a)) {}              \
  };                                                                           \
  template <class... I> requires (sizeof...(I) >= 1)                           \
  name(I...) -> name<sizeof...(I)>;

MD_ARRAY_ALIAS(darray, double)
MD_ARRAY_ALIAS(farray, float)
MD_ARRAY_ALIAS(iarray, int)
#undef MD_ARRAY_ALIAS

template <std::size_t R> using dview   = view<double, R>;
template <std::size_t R> using cdview  = view<const double, R>;
template <std::size_t R> using fview   = view<float, R>;
template <std::size_t R> using cfview  = view<const float, R>;
template <std::size_t R> using iview   = view<int, R>;
template <std::size_t R> using ciview  = view<const int, R>;

template <index_t... E> using dsarray = sarray<double, E...>;
template <index_t... E> using fsarray = sarray<float, E...>;
template <index_t... E> using isarray = sarray<int, E...>;
template <index_t... E> using dsview  = sview<double, E...>;
template <index_t... E> using cdsview = sview<const double, E...>;
template <index_t... E> using isview  = sview<int, E...>;
template <index_t... E> using cisview = sview<const int, E...>;

}  // namespace md

// Structured-binding support for rank-1 md::sarray.
template <class T, md::index_t N>
struct std::tuple_size<md::sarray<T, N>>
    : std::integral_constant<std::size_t, std::size_t(N)> {};
template <std::size_t I, class T, md::index_t N>
struct std::tuple_element<I, md::sarray<T, N>> { using type = T; };

#endif  // __cpp_lib_mdspan
#endif  // __has_include(<mdspan>)
