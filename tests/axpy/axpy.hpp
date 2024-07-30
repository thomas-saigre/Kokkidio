#include <Kokkidio.hpp>

namespace Kokkidio
{

#define KOKKIDIO_AXPY_ARGS \
	ArrayXs& z, scalar a, const ArrayXs& x, const ArrayXs& y, Index nRuns



namespace unif
{

enum class Kernel {
	cstyle,
	kokkos,
	kokkidio,
};

template<Target, Kernel>
void axpy(KOKKIDIO_AXPY_ARGS);

} // namespace unif



namespace cpu
{

enum class Kernel {
	cstyle_sequential,
	cstyle_parallel,
	eigen_sequential,
	eigen_parallel
};

template<Target, Kernel>
void axpy(KOKKIDIO_AXPY_ARGS);

} // namespace cpu



namespace gpu
{

enum class Kernel {
	cstyle,
};

template<Target, Kernel>
void axpy(KOKKIDIO_AXPY_ARGS);

} // namespace gpu



} // namespace Kokkidio
