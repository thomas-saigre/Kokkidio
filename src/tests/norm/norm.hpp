#ifndef KOKKIDIO_DOTPRODUCT_HPP
#define KOKKIDIO_DOTPRODUCT_HPP


#include <Kokkidio.hpp>

namespace Kokkidio
{

namespace cpu
{

enum class Kernel {
	cstyle_seq,
	cstyle_par,
	eigen_seq,
	eigen_par,
};

template<Target target, Kernel k>
scalar norm(const MatrixXs& m, int nRuns);

} // namespace cpu

namespace gpu
{

enum class Kernel {
	cstyle_blockbuf,
};

template<Target target, Kernel k>
scalar norm(const MatrixXs& m, int nRuns);

} // namespace gpu


namespace unif
{

enum class Kernel {
	cstyle,
	kokkidio_index,
	kokkidio_range,
};

template<Target target, Kernel k>
scalar norm(const MatrixXs& m, int nRuns);

} // namespace unif


} // namespace Kokkidio


#endif