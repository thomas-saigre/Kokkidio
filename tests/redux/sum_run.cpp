#include "bench/runAndTime.hpp"
#include "bench/parseOpts.hpp"

#include "redux.hpp"

namespace Kokkidio
{

KOKKIDIO_FUNC_WRAPPER(sum_gpu , gpu ::sum)
KOKKIDIO_FUNC_WRAPPER(sum_cpu , cpu ::sum)

void runSum( const BenchOpts& b ){
	if ( !b.gnuplot ){
		std::cout << "Running redux benchmark: sum...\n";
	}
	ArrayXXs arr (b.nRows, b.nCols);
	// #define TEST_INITIALIZATION 
	#ifdef TEST_INITIALIZATION
	/* create a predictable result */
	arr = 1;
	#else
	/* Initialize the matrices with random values */
	arr.setRandom();
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
		// ArrayXsMap resMap { results.data(), static_cast<Index>( results.size() ) };
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
	// using uK = unif::Kernel;
	/* Run on GPU */
	#ifndef KOKKIDIO_CPU_ONLY
	if ( b.target != "cpu" ){
		resetOpts();
		// runAndTime<dot_unif, T::device, uK,
		// 	uK::colwise_manual_stackbuf, // first one is for warmup
		// 	uK::colwise_manual_stackbuf,
		// 	uK::colwise_manual_nobuf,
		// 	uK::colwise_eigen,
		// 	uK::ranged,
		// 	uK::ranged_for_each,
		// 	uK::colwise_eigen_merged,
		// 	uK::ranged_for_each_merged
		// >( opts, pass, m1, m2, nRuns );
		#ifdef KOKKIDIO_USE_CUDAHIP
		using gK = gpu::Kernel;
		opts.groupComment = "native";
		opts.skipWarmup = false;
		runAndTime<sum_gpu, T::device, gK
			, gK::sum0_dyn // first one is for warmup
			, gK::sum0_dyn
			, gK::sum0_fixed
			, gK::sum6
			, gK::sum6_cbranch
			, gK::sum_generic
		>( opts, pass, arr, b.nRuns );
		#endif
	}
	#endif

	if ( b.target != "gpu" && b.nCols * b.nRuns <= 25e8 ){
		/* Run on CPU */
		resetOpts();
	// 	runAndTime<dot_unif, T::host, uK,
	// 		uK::colwise_manual_stackbuf, // first one is for warmup
	// 		uK::colwise_manual_stackbuf,
	// 		uK::colwise_manual_nobuf,
	// 		uK::colwise_eigen,
	// 		uK::ranged,
	// 		uK::ranged_for_each,
	// 		uK::colwise_eigen_merged,
	// 		uK::ranged_for_each_merged
	// 	>( opts, pass, m1, m2, nRuns );

		using cK = cpu::Kernel;
		opts.groupComment = "native";
		opts.skipWarmup = false;
		runAndTime<sum_cpu, T::host, cK
			, cK::eigen_with_local_var // first one is for warmup
			, cK::seq
			, cK::manual_global_var_only
			, cK::manual_with_local_var
			, cK::eigen_global_var_only
			, cK::eigen_with_local_var
		>( opts, pass, arr, b.nRuns );
	}

	if (!b.gnuplot){
		std::cout
			<< "Sum reduction results:\n" << resMap() << '\n'
			<< "Sum reduction: Finished runs.\n\n";
	}
}

} // namespace Kokkidio
