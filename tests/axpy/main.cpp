#include "bench/runAndTime.hpp"
#include "bench/parseOpts.hpp"

#include "axpy.hpp"

namespace Kokkidio
{

KOKKIDIO_FUNC_WRAPPER(axpy_unif, unif::axpy)
KOKKIDIO_FUNC_WRAPPER(axpy_cpu ,  cpu::axpy)
KOKKIDIO_FUNC_WRAPPER(axpy_gpu ,  gpu::axpy)


void run_axpy(const BenchOpts b){
	if ( !b.gnuplot ){
		std::cout << "Running saxpy/daxpy benchmark...\n";
	}

	scalar a, z_correct;

	ArrayXs x ( std::max(b.nRows, b.nCols) ), y, z;
	y.resizeLike(x);
	z.resizeLike(x);

	Array3s randVals;
	randVals.setRandom();
	a = randVals[0];
	x = randVals[1];
	y = randVals[2];

	z_correct = a * x[0] + y[0];

	auto pass = [&](){
		/* mapping to the results so that we can check that they're all equal */
		bool same { z.isApproxToConstant(z_correct, epsilon) };
		if ( !same ){
			std::cerr.precision(16);
			std::cerr << "z:\n";
			if (z.size() < 30){
				std::cerr << z;
			} else {
				std::cerr
					<< z.head(3)
					<< "\n...\n"
					<< z.tail(3)
				;
			}
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
		runAndTime<axpy_unif, T::device, uK
			, uK::cstyle // first one is for warmup
			, uK::cstyle
			, uK::kokkos
			, uK::kokkidio
		>( opts, pass, z, a, x, y, b.nRuns );

		using gK = gpu::Kernel;
		opts.groupComment = "native";
		opts.skipWarmup = true;
		// opts.skipWarmup = false;
		runAndTime<axpy_gpu, T::device, gK
			// , gK::cstyle
			, gK::cstyle
		>( opts, pass, z, a, x, y, b.nRuns );
	}
	#endif

	if ( b.target != "gpu" && (z.size() <= 1000 * 1000 * 1000 || b.nRuns <= 500) ){
		/* Run on CPU */
		resetOpts();
		runAndTime<axpy_unif, T::host, uK
			, uK::cstyle // first one is for warmup
			, uK::cstyle
			, uK::kokkos
			, uK::kokkidio
		>( opts, pass, z, a, x, y, b.nRuns );

		using cK = cpu::Kernel;
		opts.groupComment = "native";
		opts.skipWarmup = true;
		runAndTime<axpy_cpu, T::host, cK
			// , cK::cstyle_parallel // warmup is skipped
			, cK::cstyle_sequential
			, cK::cstyle_parallel
			, cK::eigen_sequential
			, cK::eigen_parallel
		>( opts, pass, z, a, x, y, b.nRuns );
	}

	if (!b.gnuplot){
		std::cout
			<< "saxpy/daxpy result:\n" << z_correct << '\n'
			<< "saxpy/daxpy: Finished runs.\n\n";
	}
}

} // namespace Kokkidio

int main(int argc, char** argv){

	Kokkos::ScopeGuard guard(argc, argv);

	Kokkidio::BenchOpts b;
	if ( auto exitCode = parseOpts(b, argc, argv) ){
		exit( exitCode.value() );
	}
	Kokkidio::run_axpy(b);

	return 0;
}
