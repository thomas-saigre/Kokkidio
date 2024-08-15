#include <Kokkidio.hpp>
#include <sstream>
#include <iostream>


template<Kokkidio::Target target>
void colmult(int nRows, int nCols);

int main(int argc, char** argv){
	Kokkos::ScopeGuard guard(argc, argv);

	auto ins = [&](int i, auto& var){
		if ( argc > i ){
			std::stringstream sstr {argv[i]};
			sstr >> var;
		}
	};
	int nRows {4}, nCols {1000};
	ins(1, nRows);
	ins(2, nCols);
	std::cout << "Using matrix size " << nRows << " x " << nCols << '\n';

	using T = Kokkidio::Target;

	colmult<T::device>(nRows, nCols);
	colmult<T::host>(nRows, nCols);

	return 0;
}
