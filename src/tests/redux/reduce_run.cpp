#include "runAndTime.hpp"
#include "parseOpts.hpp"

#include "redux.hpp"

namespace Kokkidio
{

KOKKIDIO_FUNC_WRAPPER(reduce_gpu , gpu::reduce)
KOKKIDIO_FUNC_WRAPPER(reduce_cpu , cpu::reduce)

void runRedux( const BenchOpts& b ){
	if ( !b.gnuplot ){
		std::cout << "Running redux benchmark: generic...\n";
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
	std::vector<scalar> results_cpu, results_gpu;
	results_cpu.reserve(4);
	results_gpu.reserve(4);

	/* mapping to the results so that we can check that they're all equal */
	auto resMap = [&](const std::vector<scalar>& res) -> ArrayXsCMap {
		return { res.data(), static_cast<Index>( res.size() ) };
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
		#ifdef KOKKIDIO_USE_CUDAHIP
		using gK = gpu::Reduction;
		opts.groupComment = "native";
		opts.skipWarmup = false;
		runAndTime<reduce_gpu, T::device, gK
			, gK::sum // first one is for warmup
			, gK::sum
			, gK::prod
			, gK::min
			, gK::max
		>(
			opts, [&](scalar result){
				results_gpu.push_back(result);
				return true;
			}, arr, b.nRuns
		);
		#endif
	}
	#endif

	if ( b.target != "gpu" && b.nCols * b.nRuns <= 25e8 ){
		/* Run on CPU */
		resetOpts();

		using cK = cpu::Reduction;
		opts.groupComment = "native";
		opts.skipWarmup = false;
		runAndTime<reduce_cpu, T::host, cK
			, cK::sum // first one is for warmup
			, cK::sum
			, cK::prod
			, cK::min
			, cK::max
		>(
			opts, [&](scalar result){
				/* two warmup steps are performed on the GPU, 
				 * so to match the results, we copy the first. */
				static bool first {true};
				if (first){
					first = false;
					results_cpu.push_back(result);
				}
				results_cpu.push_back(result);
				return true;
			}, arr, b.nRuns
		);
	}

	auto bothResults = [&](){
		ArrayXXs res (
			std::max( results_cpu.size(), results_gpu.size() ),
			2
		);
		res = 0;
		if ( b.target != "gpu" ) res.col(0) << resMap(results_cpu);
		if ( b.target != "cpu" ) res.col(1) << resMap(results_gpu);
		return res;
	};

	if ( b.target == "all" ){
		if ( ! resMap(results_gpu).isApprox( resMap(results_cpu), epsilon ) ){
			std::cerr.precision(16);
			std::cerr
				<< "Diverging results!\nCPU|GPU\n"
				<< bothResults()
				<< '\n';
		}
	}

	if (!b.gnuplot){
		std::cout
			<< "Redux results:\n" << bothResults() << '\n'
			<< "Redux: Finished runs.\n\n";
	}
}

} // namespace Kokkidio
