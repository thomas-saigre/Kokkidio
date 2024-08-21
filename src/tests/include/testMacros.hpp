#ifndef KOKKIDIO_TESTMACROS_HPP
#define KOKKIDIO_TESTMACROS_HPP

// #define KOKKIDIO_RUN_ALL_TESTS

#ifdef KOKKIDIO_RUN_ALL_TESTS
#define KRUN_IF_ALL(...) __VA_ARGS__
#else
#define KRUN_IF_ALL(...)
#endif

#endif