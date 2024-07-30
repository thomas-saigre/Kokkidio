#include "redux.hpp"
#include "bench/parseOpts.hpp"


namespace Kokkidio
{

void runSum( const BenchOpts& b );

void runRedux( const BenchOpts& b );

} // namespace Kokkidio


int main(int argc, char** argv){

	std::string exec {"sum"};

	auto parseExec = [&](CLI::App& app){
		app.add_option(
			"-x, --exec", exec,
			"The reduction bench to run (sum|gen)"
		)->check(
			CLI::IsMember( {"sum", "gen"}, CLI::ignore_case )
		);
	};
	Kokkidio::BenchOpts b;
	if ( auto exitCode = parseOpts(b, argc, argv, parseExec) ){
		exit( exitCode.value() );
	}

	if ( exec == "sum" ){
		Kokkidio::runSum(b);
	} else 
	if ( exec == "gen" ){
		Kokkidio::runRedux(b);
	}

	return 0;
}
