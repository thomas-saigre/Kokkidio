#include "norm.hpp"
#include <cassert>

namespace Kokkidio::cpu
{

using K = Kernel;
constexpr Target host { Target::host };

constexpr scalar smin { std::numeric_limits<scalar>::lowest() };

/* Sequential CPU calculation */
template<>
scalar norm<host, K::seq_cstyle>(const MatrixXs& mat, int nRuns){
	scalar result {0};

	Index
		nRows {mat.rows()},
		nCols {mat.cols()};

	const scalar* mat_ptr { mat.data() };

	for (volatile int run = 0; run < nRuns; ++run){
		result = smin;
		for (int j = 0; j < nCols; ++j){
			scalar norm {0};
			for (int i = 0; i < nRows; ++i){
				int idx = j * nRows + i;
				norm += mat_ptr[idx] * mat_ptr[idx];
			}
			result = std::max( result, sqrt(norm) );
		}
	}
	return result;
}

template<>
scalar norm<host, K::seq_eigen>(const MatrixXs& mat, int nRuns){
	scalar result {smin};

	for (volatile int run = 0; run < nRuns; ++run){
		result = mat.colwise().norm().maxCoeff();
	}
	return result;
}

template<>
scalar norm<host, K::par_cstyle>(const MatrixXs& mat, int nRuns){
	scalar result {smin};

	Index
		nRows {mat.rows()},
		nCols {mat.cols()};

	const scalar* mat_ptr { mat.data() };

	for (volatile int run = 0; run < nRuns; ++run){
		result = smin;
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			scalar result_thread {0};
			KOKKIDIO_OMP_PRAGMA(for)
			for (int j = 0; j < nCols; ++j){
				scalar norm {0};
				for (int i = 0; i < nRows; ++i){
					int idx = j * nRows + i;
					norm += mat_ptr[idx] * mat_ptr[idx];
				}
				result_thread = std::max( result_thread, sqrt(norm) );
			}
			KOKKIDIO_OMP_PRAGMA(for reduction (max:result))
			for (int i=0; i<omp_get_num_threads(); ++i){
				result = std::max( result, result_thread );
			}
		}
	}
	return result;
}

template<>
scalar norm<host, K::par_eigen>(const MatrixXs& mat, int nRuns){
	scalar result {smin};

	for (volatile int run = 0; run < nRuns; ++run){
		result = smin;
		KOKKIDIO_OMP_PRAGMA(parallel)
		{
			auto [colBeg, nCols] = ompSegment(mat.cols()).values;
			scalar result_thread =
				mat.middleCols(colBeg, nCols).colwise().norm().maxCoeff();

			KOKKIDIO_OMP_PRAGMA(for reduction (max:result))
			for (int i=0; i<omp_get_num_threads(); ++i){
				result = std::max( result, result_thread );
			}
		}
	}
	return result;
}

} // namespace Kokkidio::cpu
