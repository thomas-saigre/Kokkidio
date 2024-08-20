#ifndef KOKKIDIO_DOTPRODUCT_HPP
#define KOKKIDIO_DOTPRODUCT_HPP


#include <Kokkidio.hpp>

namespace Kokkidio
{

namespace cpu
{

enum class Kernel {
	seq_cstyle,
	seq_eigen,
	// seq_manual,
	// seq_eigen_colwise,
	// seq_eigen_arrProd,
	par_cstyle,
	par_eigen,
	// par_manual,
	// par_eigen_colwise,
	// par_eigen_colwise_ranged,
	// par_eigen_arrProd,
	// par_eigen_arrProd_ranged,
};

template<Target target, Kernel k>
scalar norm(const MatrixXs& m, int nRuns);

} // namespace cpu

namespace gpu
{

enum class Kernel {
	cstyle_blockbuf,
	// eigen_colwise,
	// eigen_colwise_merged,
	// cstyle_blockbuf,
};

template<Target target, Kernel k>
scalar norm(const MatrixXs& m, int nRuns);

} // namespace gpu


namespace unif
{

enum class Kernel {
	cstyle,
	// cstyle_nobuf,
	// cstyle_stackbuf,
	eigen_colwise,
	// eigen_colwise_merged,
	eigen_ranged,
	// eigen_ranged_chunks,
	// eigen_ranged_for_each,
	// eigen_ranged_for_each_merged,
};

template<Target target, Kernel k>
scalar norm(const MatrixXs& m, int nRuns);

} // namespace unif


} // namespace Kokkidio


#endif