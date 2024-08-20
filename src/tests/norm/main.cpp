#include "runAndTime.hpp"
#include "parseOpts.hpp"

#include "norm.hpp"

namespace Kokkidio
{

KOKKIDIO_FUNC_WRAPPER(norm_gpu , gpu ::norm)
KOKKIDIO_FUNC_WRAPPER(norm_cpu , cpu ::norm)
KOKKIDIO_FUNC_WRAPPER(norm_unif, unif::norm)

/* computes the magnitude/norm of vectors and finds the largest one */
void runNorm(const BenchOpts b){
	if ( !b.gnuplot ){
		std::cout << "Running norm benchmark...\n";
	}
	/* Initialize a matrix with random values */
	MatrixXs mat = MatrixXs::Random(b.nRows, b.nCols);

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
		runAndTime<norm_unif, T::device, uK
			, uK::eigen_ranged // first one is for warmup
			, uK::cstyle
			, uK::eigen_colwise
			, uK::eigen_ranged
		>( opts, pass, mat, b.nRuns );
		#ifdef KOKKIDIO_USE_CUDAHIP
		using gK = gpu::Kernel;
		opts.groupComment = "native";
		opts.skipWarmup = true;
		runAndTime<norm_gpu, T::device, gK
			, gK::cstyle_blockbuf
		>( opts, pass, mat, b.nRuns );
		#endif
	}
	#endif

	if ( b.target != "gpu" && b.nCols * b.nRuns <= 25e8 ){
		/* Run on CPU */
		resetOpts();
		runAndTime<norm_unif, T::host, uK
			, uK::eigen_ranged // first one is for warmup
			, uK::cstyle
			, uK::eigen_colwise
			, uK::eigen_ranged
		>( opts, pass, mat, b.nRuns );

		using cK = cpu::Kernel;
		opts.groupComment = "native";
		opts.skipWarmup = true;
		runAndTime<norm_cpu, T::host, cK
			, cK::seq_cstyle
			, cK::seq_eigen
			, cK::par_cstyle
			, cK::par_eigen
		>( opts, pass, mat, b.nRuns );
	}

	if (!b.gnuplot){
		std::cout
			<< "Norm maxima:\n" << resMap() << '\n'
			<< "Norm: Finished runs.\n\n";
	}
}

} // namespace Kokkidio

int main(int argc, char** argv){

	Kokkos::ScopeGuard guard(argc, argv);

	Kokkidio::BenchOpts b;
	b.nRows = 4;
	if ( auto exitCode = parseOpts(b, argc, argv) ){
		exit( exitCode.value() );
	}
	Kokkidio::runNorm(b);

	return 0;
}
