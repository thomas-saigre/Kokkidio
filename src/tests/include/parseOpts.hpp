#ifndef KOKKIDIO_PARSE_OPTS_HPP
#define KOKKIDIO_PARSE_OPTS_HPP

#include "CLI11.hpp"
#include <optional>

namespace Kokkidio
{

struct BenchOpts {
	std::string target {"all"};
	long
		nRuns {1},
		nRows {1},
		nCols {512};
	bool gnuplot {false};
};

inline void doNothing(CLI::App&){}

template<typename Func = void(*)(CLI::App&)>
std::optional<int> parseOpts(
	BenchOpts& opts,
	int argc, char** argv,
	Func&& parseExtra = doNothing
){
	CLI::App app {"Runner for Kokkidio benchmarks"};
	argv = app.ensure_utf8(argv);

	app.add_option(
		"-t,--target", opts.target, "Which target to run on (cpu|gpu|all/both)"
	)->check(
		CLI::IsMember( {"cpu", "gpu", "all", "both"}, CLI::ignore_case )
	);

	std::vector<long> size;
	app.add_option(
		"-s,--size", size,
		"The number of elements to use. "
		"If one argument is provided, it is used as the number of columns. "
		"If two arguments are provided, they are interpreted as rows x cols."
	)->expected(1,2)->check(CLI::PositiveNumber);

	app.add_option(
		"-r,--runs", opts.nRuns, "The number of repetitions"
	)->check(CLI::PositiveNumber);

	bool gnuplot {false};
	app.add_flag(
		"-g,--gnuplot", opts.gnuplot,
		"Whether to format output for piping to gnuplot"
	);

	parseExtra(app);

	CLI11_PARSE(app, argc, argv);

	if ( size.size() > 1 ){
		opts.nRows = size[0];
		opts.nCols = size[1];
	} else
	if ( size.size() > 0 ){
		opts.nCols = size[0];
	}

	if (!gnuplot){
		std::cout << "Using "
			<< opts.nRows << " rows, "
			<< opts.nCols << " columns, "
			<< opts.nRuns << " iterations, "
			<< "and target: " << opts.target 
			<< ".\n";
	}

	return std::nullopt;
}

} // namespace Kokkidio

#endif
