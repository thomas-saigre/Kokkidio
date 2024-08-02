
/* we want the unified functions to compile on all backends. */
#define KOKKIDIO_AXPY_TARGET Target::host
#include "axpy_unif.in"
