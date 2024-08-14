#include <Kokkidio.hpp>
#include <iostream>

#include "magic_enum.hpp"


template<typename ViewMapType>
void printFirstN( int nPrint, const ViewMapType& a, const ViewMapType& b );


int main(int argc, char** argv){
	Kokkos::ScopeGuard guard(argc, argv);

	using namespace Kokkidio;
	std::cout << "DefaultTarget: " << magic_enum::enum_name(DefaultTarget) <<'\n';
	constexpr Target target {DefaultTarget};
	// constexpr Target target {Target::host};

	int nRows {4}, nCols {5};

	/* create and set the input matrices */
	using MatrixView = DualViewMap<Eigen::MatrixXd, target>;
	MatrixView a {nRows, nCols}, b {nRows, nCols};
	// b.resizeLike(a);

	a.map_host().setRandom();
	b.map_host().setRandom();

	a.copyToTarget();
	b.copyToTarget();

	double result = 0;
	/* perform parallel computation and reduction (2D -> column range) */
	Kokkidio::parallel_reduce<target>(
		a.cols(),
		// nCols,
		KOKKOS_LAMBDA(ParallelRange<target> rng, double& sum){
			// sum += ( rng(a).transpose() * rng(b) ).trace(); // trace = sum of the diagonal
			/* equivalent: sum of coefficient-wise products */
			printf("isAlloc (addr|bool), a: %p|%s, b: %p|%s\n"
				, (void*) a.view_target().data()
				, a.view_target().is_allocated() ? "true" : "false"
				, (void*) b.view_target().data()
				, b.isAlloc_target() ? "true" : "false"
			);
			// sum += ( rng(a).array() * rng(b).array() ).sum();
			// sum += ( rng( a.map_target() ).array() * rng( b.map_target() ).array() ).sum();
			sum += (
				detail::colRange( rng.get(), a.map_target() ).array() *
				detail::colRange( rng.get(), b.map_target() ).array()
			).sum();
		},
		redux::sum(result)
	);

	std::cout
		<< "Result: " << result
		// << ", expected: " << ( a.map_host().transpose() * b.map_host() ).trace()
		<< ", expected: " << ( a.map_host().array() * b.map_host().array() ).sum()
		<< '\n';
	printFirstN(5, a, b);

	return 0;
}





template<typename ViewMapType>
void printFirstN( int nPrint, const ViewMapType& a, const ViewMapType& b ){
	int nCols { a.cols() };
	nPrint = std::min(nPrint, nCols);

	/* col buffer for printing */
	Eigen::MatrixXd colBuf ( a.rows(), 2 );
	int precision {4}, opts {0};
	Eigen::IOFormat fmt( precision, opts, " * ", " + \n", "(", ")" );

	std::stringstream str;
	// str << std::fixed;
	for (int i{0}; i<nPrint; ++i){
		colBuf << a.map_host().col(i), b.map_host().col(i);
		str
			<< colBuf.format(fmt) << " = " 
			<< colBuf.col(0).dot(colBuf.col(1))
			<< '\n';
	}
	if (nPrint < nCols){
		str << "...\n";
	}
	std::cout << "Check:\n" << str.str();
}
