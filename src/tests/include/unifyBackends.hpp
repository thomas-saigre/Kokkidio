#ifndef KOKKIDIO_UNIFYBACKENDS_HPP
#define KOKKIDIO_UNIFYBACKENDS_HPP

#ifndef KOKKIDIO_UNIFYBACKENDS_PUBLIC_HEADER
#define KOKKIDIO_UNIFYBACKENDS_PUBLIC_HEADER
#endif

#ifndef KOKKIDIO_PUBLIC_HEADER
#define KOKKIDIO_PUBLIC_HEADER
#include "Kokkidio/syclify_macros.hpp"
#undef KOKKIDIO_PUBLIC_HEADER
#endif


#ifdef KOKKIDIO_USE_CUDAHIP
#include "unify_cuda_hip.hpp"
#elif defined(KOKKIDIO_USE_OMPT)
#include "unify_omptarget.hpp"
#endif

#undef KOKKIDIO_UNIFYBACKENDS_PUBLIC_HEADER

#endif