#include "bench/runAndTime.hpp"
#include "bench/parseOpts.hpp"

#include "redux.hpp"

namespace Kokkidio::cpu
{

constexpr Target host { Target::host };
using K = Kernel;

template<>
scalar sum<host, K::seq>(const ArrayXXs& values, int nRuns){
	scalar sum_g {0};
	for (int iter {0}; iter<nRuns; ++iter){
		sum_g = 0;
		sum_g = values.sum();
	}
	return sum_g;
}

template<>
scalar sum<host, K::manual_global_var_only>(const ArrayXXs& values, int nRuns){
	scalar sum_g {0};
	for (int iter {0}; iter<nRuns; ++iter){
		sum_g = 0;
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			KOKKIDIO_OMP_PRAGMA(for reduction (+:sum_g))
			for (Index i = 0; i < values.size(); ++i){
				sum_g += values.data()[i];
			}
		}
	}
	return sum_g;
}

template<>
scalar sum<host, K::manual_with_local_var>(const ArrayXXs& values, int nRuns){
	scalar sum_g {0};
	for (int iter {0}; iter<nRuns; ++iter){
		sum_g = 0;
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			scalar sum_l {0};
			KOKKIDIO_OMP_PRAGMA(for)
			for (Index i = 0; i < values.size(); ++i){
				sum_l += values.data()[i];
			}

			KOKKIDIO_OMP_PRAGMA(atomic)
			sum_g += sum_l;
		}
	}
	return sum_g;
}

template<>
scalar sum<host, K::eigen_global_var_only>(const ArrayXXs& values, int nRuns){
	scalar sum_g {0};
	for (int iter {0}; iter<nRuns; ++iter){
		sum_g = 0;
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			Kokkidio::EigenRange<host> rng { ompSegment( values.cols() ) };

			KOKKIDIO_OMP_PRAGMA(atomic)
			sum_g += rng(values).sum();
		}
		// sum_g = values.sum();
	}
	return sum_g;
}

template<>
scalar sum<host, K::eigen_with_local_var>(const ArrayXXs& values, int nRuns){
	scalar sum_g {0};
	for (int iter {0}; iter<nRuns; ++iter){
		sum_g = 0;
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			Kokkidio::EigenRange<host> rng { ompSegment( values.cols() ) };
			scalar sum_l { rng(values).sum() };

			KOKKIDIO_OMP_PRAGMA(atomic)
			sum_g += sum_l;
		}
		// sum_g = values.sum();
	}
	return sum_g;
}

} // namespace Kokkidio::cpu
