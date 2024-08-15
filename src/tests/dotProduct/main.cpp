#include "runAndTime.hpp"
#include "parseOpts.hpp"

#include "dotProduct.hpp"

namespace Kokkidio
{

KOKKIDIO_FUNC_WRAPPER(dot_gpu , gpu ::dotProduct)
KOKKIDIO_FUNC_WRAPPER(dot_cpu , cpu ::dotProduct)
KOKKIDIO_FUNC_WRAPPER(dot_unif, unif::dotProduct)


void runDot(const BenchOpts b){
	if ( !b.gnuplot ){
		std::cout << "Running dot product benchmark...\n";
	}
	#ifdef TEST_INITIALIZATION
	/* Test whether everything is okay */
	MatrixXs m1 (b.nRows, b.nCols), m2 (b.nRows, b.nCols);
	m1.array() = 1;
	m2 = m1;
	m2.array().row(0) = -1;
	#else
	/* Initialize the matrices with random values */
	MatrixXs m1 = MatrixXs::Random(b.nRows, b.nCols);
	MatrixXs m2 = MatrixXs::Random(b.nRows, b.nCols);
	#endif

	/* let's store the results of each run,
	 * so that we can compare them and report when they're not equal,
	 * rather than having to check that manually. */
	std::vector<scalar> results;
	results.reserve(16);

	/* mapping to the results so that we can check that they're all equal */
	auto resMap = [&]() -> ArrayXsMap {
		return { results.data(), static_cast<Index>( results.size() ) };
	};

	auto pass = [&](scalar result){
		results.push_back( result );
		/* mapping to the results so that we can check that they're all equal */
		bool same { resMap().isApproxToConstant(results[0], epsilon) };
		if ( !same ){
			std::cerr.precision(16);
			std::cerr << "Diverging results!\n" << resMap() << '\n';
		}
		return same;
	};

	RunOpts opts;
	auto resetOpts = [&](){
		opts.groupComment = "unified";
		opts.skipWarmup = false;
		opts.useGnuplot = b.gnuplot;
	};

	using T = Target;
	using uK = unif::Kernel;
	/* Run on GPU */
	#ifndef KOKKIDIO_CPU_ONLY
	if ( b.target != "cpu" ){
		resetOpts();
		runAndTime<dot_unif, T::device, uK
			, uK::eigen_ranged // first one is for warmup
			, uK::cstyle_stackbuf
			, uK::cstyle_nobuf
			, uK::eigen_colwise
			, uK::eigen_ranged
			, uK::eigen_ranged_chunks
			// , uK::eigen_ranged_for_each
			, uK::eigen_colwise_merged
			, uK::eigen_ranged_for_each_merged
		>( opts, pass, m1, m2, b.nRuns );
		#ifdef KOKKIDIO_USE_CUDAHIP
		using gK = gpu::Kernel;
		opts.groupComment = "native";
		opts.skipWarmup = true;
		runAndTime<dot_gpu, T::device, gK
			// , gK::eigen_colwise_merged
			// , gK::eigen_colwise
			, gK::cstyle_blockbuf
		>( opts, pass, m1, m2, b.nRuns );
		#endif
	}
	#endif

	if ( b.target != "gpu" && b.nCols * b.nRuns <= 25e8 ){
		/* Run on CPU */
		resetOpts();
		runAndTime<dot_unif, T::host, uK
			, uK::eigen_ranged // first one is for warmup
			, uK::cstyle_stackbuf
			, uK::cstyle_nobuf
			, uK::eigen_colwise
			, uK::eigen_ranged
			, uK::eigen_ranged_chunks
			// , uK::ranged_for_each
			, uK::eigen_colwise_merged
			, uK::eigen_ranged_for_each_merged
		>( opts, pass, m1, m2, b.nRuns );

		using cK = cpu::Kernel;
		opts.groupComment = "native";
		opts.skipWarmup = true;
		runAndTime<dot_cpu, T::host, cK
			, cK::seq_cstyle
			, cK::seq_eigen_colwise
			, cK::seq_eigen_arrProd
			, cK::par_cstyle
			, cK::par_eigen_colwise
			// , cK::par_colwise_ranged
			, cK::par_eigen_arrProd
			// , cK::par_arrProd_ranged
		>( opts, pass, m1, m2, b.nRuns );
	}

	if (!b.gnuplot){
		std::cout
			<< "DotProduct sums:\n" << resMap() << '\n'
			<< "DotProduct: Finished runs.\n\n";
	}
}

} // namespace Kokkidio

int main(int argc, char** argv){

	Kokkos::ScopeGuard guard(argc, argv);

	Kokkidio::BenchOpts b;
	if ( auto exitCode = parseOpts(b, argc, argv) ){
		exit( exitCode.value() );
	}
	Kokkidio::runDot(b);

	return 0;
}
