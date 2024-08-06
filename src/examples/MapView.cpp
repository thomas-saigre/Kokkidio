#include <Kokkidio.hpp>


int main(int argc, char** argv){
	Kokkos::ScopeGuard guard(argc, argv);

	using namespace Kokkidio;
	int nRows {10}, nCols {20};

	/* existing Eigen object */
	Eigen::ArrayXXd eigenArray {nRows, nCols};

	/* Create MapView using a constructor or factory function.
	 * Deduces Eigen type, and uses default target */
	MapView mv1 {eigenArray};
	auto mv2 = mapView(eigenArray);

	/* Create MapView using factory function for specific target,
	 * while deducing Eigen type */
	auto mv3 = mapView<Target::host>(eigenArray);

	/* Create MapView using size parameters. 
	 * ArrayXXd is dynamically sized in both dimensions, 
	 * so two parameters are required */
	MapView<Eigen::ArrayXXd> mv4 {nRows, nCols};

	/* ArrayXd is a column vector, so only rows are required */
	MapView<Eigen::ArrayXd> mv5 {nRows};

	/* Array3d is a fixed size type, so no parameters are required */
	MapView<Eigen::Array3d> mv6;

	return 0;
}