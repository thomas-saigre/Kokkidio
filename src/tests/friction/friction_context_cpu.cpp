// #if defined(__CUDACC__)
// #warning "This file should not be seen by nvcc. Check inclusion order and assumptions about #defines"
// #endif

#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
#error "When using nvcc for host compilation, EIGEN_NO_CUDA must be defined!"
#endif

#undef EIGEN_DONT_VECTORIZE

/* we want the unified functions to compile on all backends. */
#define KOKKIDIO_FRICTION_TARGET Target::host
#include "friction_context.in"