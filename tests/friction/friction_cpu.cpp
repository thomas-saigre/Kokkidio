#include "friction.hpp"

namespace Kokkidio::cpu {

constexpr Target host {Target::host};

template<>
void friction<host, Kernel::eigen_ranged_fullbuf>(
	ArrayXXs& flux_out,
	const ArrayXXs& flux_in,
	const ArrayXXs& d,
	const ArrayXXs& v,
	const ArrayXXs& n,
	int nRuns
){
	#ifndef NDEBUG
	auto assertCols = [&](const auto& arr){
		assert( flux_out.cols() == arr.cols() );
	};
	assertCols(flux_in);
	assertCols(d);
	assertCols(v);
	assertCols(n);
	#endif

	static constexpr Index bufRows {3};

	Index nColsTotal { flux_out.cols() };
	ArrayXXs buf {bufRows, nColsTotal};

	for (int it{0}; it<nRuns; ++it){
		#ifdef KOKKIDIO_OPENMP
		#pragma omp parallel
		#endif
		{
			auto [colBeg, nCols] = ompSegment(nColsTotal).values;
			detail::friction_buf3(
				buf     .middleCols(colBeg, nCols),
				flux_out.middleCols(colBeg, nCols),
				flux_in .middleCols(colBeg, nCols),
				d       .middleCols(colBeg, nCols),
				v       .middleCols(colBeg, nCols),
				n       .middleCols(colBeg, nCols)
			);
		}
	}
}



template<>
void friction<host, Kernel::cstyle>(
	ArrayXXs& flux_out,
	const ArrayXXs& flux_in,
	const ArrayXXs& d,
	const ArrayXXs& v,
	const ArrayXXs& n,
	int nRuns
){
	#ifndef NDEBUG
	auto assertCols = [&](const auto& arr){
		assert( flux_out.cols() == arr.cols() );
	};
	assertCols(flux_in);
	assertCols(d);
	assertCols(v);
	assertCols(n);
	#endif

	static constexpr Index bufRows {3};

	Index nColsTotal { flux_out.cols() };
	ArrayXXs buf {bufRows, nColsTotal};

	for (int it{0}; it<nRuns; ++it){
		using Kokkidio::detail::pow;
		using Kokkidio::detail::sqrt;
		KOKKIDIO_OMP_PRAGMA(parallel for)
		for (int i=0; i<nColsTotal; ++i){
			scalar* flux_out_d {flux_out.data()};
			const scalar
				* flux_in_d {flux_in.data()},
				* d_d {d.data()},
				* v_d {v.data()},
				* n_d {n.data()};
			scalar
				vNorm { sqrt( pow(v_d[2 * i], 2) + pow(v_d[2 * i + 1], 2) ) },
				chezyFac = phys::g * pow(n_d[i], 2) / pow(d_d[i], 1./3),
				fricFac = chezyFac * vNorm;
			chezyFac /= d_d[i];

			for (Index row = 0; row<2; ++row){
				flux_out_d[3 * i + row + 1] =
				(flux_in_d[3 * i + row + 1] - fricFac * v_d[2 * i + row] ) /
				( 1 + chezyFac * ( vNorm + pow(v_d[2 * i + row], 2) / vNorm ) );
			}
		}
	}
}

} // namespace Kokkidio::cpu
