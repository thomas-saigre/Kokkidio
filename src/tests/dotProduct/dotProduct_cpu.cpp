#include "dotProduct.hpp"
#include <cassert>

namespace Kokkidio::cpu
{

using K = Kernel;
constexpr Target host { Target::host };

/* Sequential CPU calculation */
template<>
scalar dotProduct<host, K::seq_cstyle>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	scalar finalResult_cpu {0};
	for (volatile int iter = 0; iter < iterations; ++iter){
		finalResult_cpu = 0;
		const scalar
			*p1 {m1.data()},
			*p2 {m2.data()};
		Index
			cols {m1.cols()},
			rows {m1.rows()};

		for (int col = 0; col < cols; ++col){
		for (int row = 0; row < rows; ++row){
			finalResult_cpu += p1[col * rows + row] * p2[col * rows + row];
		}}
	}
	return finalResult_cpu;
}

// template<>
// scalar dotProduct<host, K::seq_manual>(
// 	const MatrixXs& m1, const MatrixXs& m2, int iterations
// ){
// 	scalar finalResult_cpu {0};
// 	for (volatile int iter = 0; iter < iterations; ++iter){
// 		finalResult_cpu = 0;
// 		for (int col = 0; col < m1.cols(); ++col){
// 		for (int row = 0; row < m1.rows(); ++row){
// 			finalResult_cpu += m1(row, col) * m2(row, col);
// 		}}
// 	}
// 	return finalResult_cpu;
// }

template<>
scalar dotProduct<host, K::seq_eigen_colwise>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	scalar finalResult_cpu {0};
	for (volatile int iter = 0; iter < iterations; ++iter){
		finalResult_cpu = 0;
		for (int col = 0; col < m1.cols(); ++col){
			finalResult_cpu += m1.col(col).dot(m2.col(col));
		}
	}
	return finalResult_cpu;
}

template<>
scalar dotProduct<host, K::seq_eigen_arrProd>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	scalar finalResult_cpu {0};
	for (volatile int iter = 0; iter < iterations; ++iter){
		finalResult_cpu = ( m1.array() * m2.array() ).sum();
	}
	return finalResult_cpu;
}

}   // namespace Kokkidio::cpu
