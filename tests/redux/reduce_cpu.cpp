#include "redux.hpp"

namespace Kokkidio::cpu
{

constexpr Target host { Target::host };
using R = Reduction;

template<>
scalar reduce<host, R::sum>(const ArrayXXs& values, int nRuns){
	scalar val_g;
	auto setNeutral = [&](){
		val_g = 0;
	};
	setNeutral();
	for (int iter {0}; iter<nRuns; ++iter){
		setNeutral();
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			Kokkidio::EigenRange<host> rng { ompSegment( values.cols() ) };
			scalar val_l { rng(values).sum() };

			KOKKIDIO_OMP_PRAGMA(atomic)
			val_g += val_l;
		}
	}
	return val_g;
}

template<>
scalar reduce<host, R::prod>(const ArrayXXs& values, int nRuns){
	scalar val_g;
	auto setNeutral = [&](){
		val_g = 1;
	};
	setNeutral();
	for (int iter {0}; iter<nRuns; ++iter){
		setNeutral();
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			Kokkidio::EigenRange<host> rng { ompSegment( values.cols() ) };
			scalar val_l { rng(values).prod() };

			KOKKIDIO_OMP_PRAGMA(atomic)
			val_g *= val_l;
		}
	}
	return val_g;
}

template<>
scalar reduce<host, R::min>(const ArrayXXs& values, int nRuns){
	scalar val_g;
	auto setNeutral = [&](){
		val_g = std::numeric_limits<scalar>::max();
	};
	setNeutral();
	for (int iter {0}; iter<nRuns; ++iter){
		setNeutral();
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			Kokkidio::EigenRange<host> rng { ompSegment( values.cols() ) };
			scalar val_l { rng(values).minCoeff() };

			KOKKIDIO_OMP_PRAGMA( for reduction (min:val_g) )
			for (int i=0; i<omp_get_num_threads(); ++i){
				val_g = std::min(val_g, val_l);
			}
		}
	}
	return val_g;
}

template<>
scalar reduce<host, R::max>(const ArrayXXs& values, int nRuns){
	scalar val_g;
	auto setNeutral = [&](){
		val_g = std::numeric_limits<scalar>::lowest();
	};
	setNeutral();
	for (int iter {0}; iter<nRuns; ++iter){
		setNeutral();
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			Kokkidio::EigenRange<host> rng { ompSegment( values.cols() ) };
			scalar val_l { rng(values).maxCoeff() };

			KOKKIDIO_OMP_PRAGMA( for reduction (max:val_g) )
			for (int i=0; i<omp_get_num_threads(); ++i){
				val_g = std::max(val_g, val_l);
			}
		}
	}
	return val_g;
}

} // namespace Kokkidio::cpu
