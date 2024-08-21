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
	eigen_seq_colwise,
	eigen_seq_arrProd,
	eigen_par_colwise,
	// eigen_par_colwise_ranged,
	eigen_par_arrProd,
	// eigen_par_arrProd_ranged,
};

template<Target target, Kernel k>
scalar dotProduct(const MatrixXs& m1, const MatrixXs& m2, int nRuns);

} // namespace cpu

namespace gpu
{

enum class Kernel {
	eigen_colwise,
	// eigen_colwise_merged,
	cstyle_blockbuf,
};

template<Target target, Kernel k>
scalar dotProduct(const MatrixXs& m1, const MatrixXs& m2, int nRuns);

} // namespace gpu


namespace unif
{

enum class Kernel {
	cstyle,
	cstyle_nobuf,
	kokkidio_index,
	kokkidio_index_merged,
	kokkidio_range,
	kokkidio_range_chunks,
	kokkidio_range_for_each,
	kokkidio_range_for_each_merged,
	kokkidio_range_trace,
};

template<Target target, Kernel k>
scalar dotProduct(const MatrixXs& m1, const MatrixXs& m2, int nRuns);

} // namespace unif


} // namespace Kokkidio

#endif