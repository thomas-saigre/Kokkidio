#include "axpy.hpp"

namespace Kokkidio::cpu
{

using K = Kernel;
constexpr Target host { Target::host };

template<>
void axpy<host, K::cstyle_seq>(KOKKIDIO_AXPY_ARGS){

	for (volatile int run = 0; run < nRuns; ++run){
		scalar* zptr { z.data() };
		const scalar
			* xptr { x.data() },
			* yptr { y.data() };

		for (int i = 0; i<z.rows(); ++i){
			zptr[i] = a * xptr[i] + yptr[i];
		}
	}
}

template<>
void axpy<host, K::cstyle_par>(KOKKIDIO_AXPY_ARGS){

	for (volatile int run = 0; run < nRuns; ++run){
		scalar* zptr { z.data() };
		const scalar
			* xptr { x.data() },
			* yptr { y.data() };

		KOKKIDIO_OMP_PRAGMA(parallel for)
		for (int i = 0; i<z.rows(); ++i){
			zptr[i] = a * xptr[i] + yptr[i];
		}
	}
}

template<>
void axpy<host, K::eigen_seq>(KOKKIDIO_AXPY_ARGS){
	for (volatile int run = 0; run < nRuns; ++run){
		z = a * x + y;
	}
}

template<>
void axpy<host, K::eigen_par>(KOKKIDIO_AXPY_ARGS){

	for (volatile int run = 0; run < nRuns; ++run){
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			auto rows = ompSegment( z.rows() );
			auto seg = [&](auto& obj){
				return obj.segment( rows.start(), rows.count() );
			};
			seg(z) = a * seg(x) + seg(y);
		}
	}
}

} // namespace Kokkidio::cpu
