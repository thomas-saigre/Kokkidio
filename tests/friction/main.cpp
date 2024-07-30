#include "bench/runAndTime.hpp"
#include "bench/parseOpts.hpp"

#include "friction.hpp"

namespace Kokkidio
{


KOKKIDIO_FUNC_WRAPPER(fric_unif, unif::friction)
KOKKIDIO_FUNC_WRAPPER(fric_cpu , cpu::friction)
KOKKIDIO_FUNC_WRAPPER(fric_gpu , gpu::friction)


// void runFric(int b.nCols, int b.nRuns, const std::string& b.target, bool b.gnuplot){
void runFric(const BenchOpts& b){
	if ( !b.gnuplot ){
		std::cout << "Running friction benchmark...\n";
	}
	scalar
		dVal {2},
		vx   {3},
		vy   {4},
		nVal {0.025};
	Array2s vCol {vx, vy};
	Array3s flux { dVal, 1, 2 };

	ArrayXXs
		flux_out { 3, b.nCols },
		flux_in  { 3, b.nCols },
		d { 1, b.nCols },
		v { 2, b.nCols },
		n { 1, b.nCols };
	n = nVal;
	d = dVal;
	v.colwise() = vCol;
	flux_in.colwise() = flux;
	flux_out = 0;

	auto pass = [&](){
		// return true;
		Array2s correctFlux {
			0.9119160834218206,
			1.865452694025437
		};
		bool isCorrect { flux_out.bottomRows(2).isApprox(
			correctFlux.replicate(1, b.nCols)
		) };
		if (!isCorrect){
			std::cerr.precision(16);
			std::cerr << "flux_out:\n";
			if (flux_out.size() < 30){
				std::cerr << flux_out;
			} else {
				std::cerr
					<< flux_out.leftCols(3)
					<< "\n...\n"
					<< flux_out.rightCols(3)
				;
			}
		}
		return isCorrect;
	};

	RunOpts opts;
	auto resetOpts = [&](){
		opts.groupComment = "unified";
		opts.skipWarmup = false;
		opts.useGnuplot = b.gnuplot;
	};

	// using T = Target;
	using K = fric::Kernel;
	/* GPU computation */
	#ifndef KOKKIDIO_CPU_ONLY
	if ( b.target != "cpu" ){
		resetOpts();
		runAndTime<fric_unif, Target::device, K
			, K::eigen_ranged_chunkbuf // first one is for warmup
			// #ifndef KOKKIDIO_USE_SYCL
			, K::cstyle
			// #endif
			, K::eigen_colwise_fullbuf
			, K::eigen_colwise_stackbuf
			, K::eigen_ranged_fullbuf
			, K::eigen_ranged_chunkbuf
			, K::context_ranged
		>(
			opts, pass,
			flux_out, flux_in, d, v, n, b.nRuns
		);

		/* non-unified */
		#ifdef KOKKIDIO_USE_CUDAHIP
		opts.groupComment = "native";
		opts.skipWarmup = true;
		runAndTime<fric_gpu, Target::device, K
			, K::cstyle
			, K::eigen_colwise_fullbuf
		>(
			opts, pass,
			flux_out, flux_in, d, v, n, b.nRuns
		);
		#endif
	}
	#endif

	if ( b.target != "gpu" && b.nCols * b.nRuns <= 25e8 ){
		resetOpts();
		runAndTime<fric_unif, Target::host, K
			, K::eigen_ranged_chunkbuf // first one is for warmup
			, K::cstyle
			, K::eigen_colwise_fullbuf // painfully slow
			, K::eigen_colwise_stackbuf // painfully slow
			, K::eigen_ranged_fullbuf
			, K::eigen_ranged_chunkbuf
			, K::context_ranged
		>(
			opts, pass,
			flux_out, flux_in, d, v, n, b.nRuns
		);
		/* non-unified */
		// #ifdef KOKKIDIO_USE_OMPT
		opts.groupComment = "native";
		opts.skipWarmup = true;
		runAndTime<fric_cpu, Target::host, K
			, K::cstyle
			// , K::eigen_ranged_fullbuf
		>(
			opts, pass,
			flux_out, flux_in, d, v, n, b.nRuns
		);
		// #endif
	}


	if (!b.gnuplot){
		std::cout << "Friction: Finished runs.\n\n";
	}
}


} // namespace Kokkidio

int main(int argc, char ** argv){

	Kokkos::ScopeGuard guard(argc, argv);

	Kokkidio::BenchOpts b;
	if ( auto exitCode = parseOpts(b, argc, argv) ){
		exit( exitCode.value() );
	}
	Kokkidio::runFric(b);
	
	return 0;
}
