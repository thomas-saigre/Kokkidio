#include <Kokkidio.hpp>
#include <iostream>

#include "magic_enum.hpp"


int main(int argc, char** argv){
	Kokkos::ScopeGuard guard(argc, argv);

	using namespace Kokkidio;
	std::cout << "DefaultTarget: " << magic_enum::enum_name(DefaultTarget) <<'\n';

	int nRows {10}, nCols {20};

	/* existing Eigen object */
	Eigen::ArrayXXd eigenArray {nRows, nCols};

	/*********************************************************/
	/* Construction */
	/*********************************************************/

	/* Create MapView using factory function for specific target,
	 * while deducing Eigen type */
	auto mv1 = mapView<Target::host>(eigenArray);

	/* Create MapView using a constructor or factory function.
	 * Deduces Eigen type, and uses default target */
	MapView mv2 {eigenArray};
	auto mv3 = mapView(eigenArray);
	/* Note: 
	 * Do not actually pass the same Eigen object to multiple MapViews!
	 * If you do, then resize() will only apply to the Eigen object
	 * and the MapView it was called on, but not the other MapViews. */


	/* Create MapView using size parameters. 
	 * ArrayXXd is dynamically sized in both dimensions, 
	 * so two parameters are required */
	MapView<Eigen::ArrayXXd> mv4 {nRows, nCols};

	/* ArrayXd is a column vector, so only rows are required */
	MapView<Eigen::ArrayXd> mv5 {nRows};

	/* Array3d is a fixed size type, so no parameters are required */
	MapView<Eigen::Array3d> mv6;


	/*********************************************************/
	/* Using MapView */
	/*********************************************************/

	/* set values on host, using Eigen's assignment operator on MapView::map() */
	mv1.map() = 1;
	/* set values on target, using Kokkos::deep_copy with MapView::view() */
	Kokkos::deep_copy(mv2.view(), 2);
	/* set values on target with parallel dispatch: */
	/* with Kokkidio::ParallelRange */
	parallel_for( mv3.cols(), KOKKOS_LAMBDA(ParallelRange<> rng){
		rng(mv3) = 3;
	});
	/* or just an integer, using the standard Kokkos-style */
	parallel_for( mv4.size(), KOKKOS_LAMBDA(int i){
		mv4.data()[i] = 4;
	});

	return 0;
}