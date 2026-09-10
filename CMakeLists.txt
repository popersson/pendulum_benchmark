# Pendulum benchmark -- the compiled implementations (C, C++, Fortran).
#
# Every language is optional: if a compiler is missing, or if the C++ compiler
# has no <mdspan>, the affected targets are skipped with a message instead of
# failing the configure step.  Flags are per language and deliberately plain --
# no -ffast-math, no -march=native, no threads -- so that all implementations
# solve exactly the same problem.  See README.md.

cmake_minimum_required(VERSION 3.20)
project(pendulum_benchmark LANGUAGES NONE)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(CheckLanguage)

# ------------------------------------------------------------------ C
check_language(C)
if(CMAKE_C_COMPILER)
    enable_language(C)
    add_executable(pendulum_c c/pendulum.c)
    target_compile_options(pendulum_c PRIVATE -O3)
    if(UNIX)
        target_link_libraries(pendulum_c PRIVATE m)
    endif()
else()
    message(STATUS "pendulum: no C compiler, skipping pendulum_c")
endif()

# ------------------------------------------------------------------ C++
check_language(CXX)
if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    include(CheckCXXSourceCompiles)

    # pendulum.cpp needs only C++20 (std::span + the static_vector header), so it
    # builds with any current compiler.  The two mdspan variants need <mdspan>,
    # which as of today means clang with libc++.
    add_executable(pendulum_cpp cpp/pendulum.cpp)
    target_compile_options(pendulum_cpp PRIVATE -O3 -fno-math-errno)
    target_include_directories(pendulum_cpp PRIVATE cpp)

    set(MDSPAN_PROBE "#include <mdspan>
#include <cstddef>
int main() {
    double a[8];
    std::mdspan<double, std::extents<std::size_t, 4, std::dynamic_extent>,
                std::layout_left> m(a, 2);
    return static_cast<int>(m[0, 1]);
}")

    set(CMAKE_REQUIRED_FLAGS "-std=c++23")
    check_cxx_source_compiles("${MDSPAN_PROBE}" HAVE_MDSPAN)
    if(NOT HAVE_MDSPAN)
        set(CMAKE_REQUIRED_FLAGS "-std=c++23 -stdlib=libc++")
        check_cxx_source_compiles("${MDSPAN_PROBE}" HAVE_MDSPAN_LIBCXX)
    endif()
    unset(CMAKE_REQUIRED_FLAGS)

    if(HAVE_MDSPAN OR HAVE_MDSPAN_LIBCXX)
        foreach(variant mdspan colref)
            add_executable(pendulum_cpp_${variant} cpp/pendulum_${variant}.cpp)
            target_compile_options(pendulum_cpp_${variant} PRIVATE -O3 -fno-math-errno)
            target_include_directories(pendulum_cpp_${variant} PRIVATE cpp)
            if(HAVE_MDSPAN_LIBCXX)
                target_compile_options(pendulum_cpp_${variant} PRIVATE -stdlib=libc++)
                target_link_options(pendulum_cpp_${variant} PRIVATE -stdlib=libc++)
            endif()
        endforeach()
        if(HAVE_MDSPAN_LIBCXX)
            target_compile_options(pendulum_cpp PRIVATE -stdlib=libc++)
            target_link_options(pendulum_cpp PRIVATE -stdlib=libc++)
        endif()
    else()
        message(STATUS "pendulum: no <mdspan> with this C++ compiler, skipping "
                       "pendulum_cpp_mdspan and pendulum_cpp_colref "
                       "(try -DCMAKE_CXX_COMPILER=clang++ with libc++ installed)")
    endif()
else()
    message(STATUS "pendulum: no C++ compiler, skipping the C++ targets")
endif()

# ------------------------------------------------------------------ Fortran
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    add_executable(pendulum_f fortran/pendulum.f90)
    target_compile_options(pendulum_f PRIVATE -O3)
    add_executable(pendulum_f_modern fortran/pendulum_modern.f90)
    target_compile_options(pendulum_f_modern PRIVATE -O3)
    set_target_properties(pendulum_f_modern PROPERTIES
                          Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
else()
    message(STATUS "pendulum: no Fortran compiler, skipping pendulum_f and pendulum_f_modern")
endif()
