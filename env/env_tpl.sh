# paths to third-party libraries or general modules go here.


# white desktop @ 575
if [[ $(echo $USER) == "psneguir" ]]; then
	export EIGEN_ROOT="/home/psneguir/Projekte/eigen-3.4.0_normal_src/build"
	# /home/psneguir/Projekte/env/sycl_workspace/llvm/build

# hlrn-gpu
elif [[ $node_name == *"glogin"* ]] || [[ $node_name == *"ggpu"* ]]; then
	# module load cmake/3.26.4
	module load cmake
	# compiler=$(command -v nvcc)
	PKGDIR="/scratch-emmy/usr/`whoami`"
	export EIGEN_ROOT="$PKGDIR/eigen/build"
	Kokkos_BASE="$PKGDIR/kokkos"
	# only Kokkos_SRC is required
	export Kokkos_SRC="$Kokkos_BASE/src/kokkos-dev"
	# Kokkos_BUILD and Kokkos_INST are optional,
	# but may be used to specify build/install directories for Kokkos.
	export Kokkos_BUILD="$Kokkos_BASE/build"
	export Kokkos_INST="$Kokkos_BASE/install"

# TU HPC
elif [[ $node_name == "frontend"* ]] || [[ $node_name == "gpu"* ]]; then
	module load cmake/3.28.1
	export EIGEN_ROOT="/scratch/wahyd/eigen/build"
	# eigen="$(realpath "$(cat "$HOME/.cmake/packages/Eigen3/"*)"/..)"

# HLRN, Intel PVC
elif [[ $node_name == "bgi"* ]]; then
	module load intel
	export EIGEN_ROOT="$HOME/pkg/eigen/build"

# lenni home office
elif [[ $node_name == "desktop-home-neon" ]]; then
	export EIGEN_ROOT="$HOME/pkg/eigen/build"
	Kokkos_BASE="$HOME/pkg/kokkos"
	# only Kokkos_SRC is required
	export Kokkos_SRC="$Kokkos_BASE/src/kokkos-dev"
	# Kokkos_BUILD and Kokkos_INST are optional,
	# but may be used to specify build/install directories for Kokkos.
	export Kokkos_BUILD="$Kokkos_BASE/build"
	export Kokkos_INST="$Kokkos_BASE/install"

else
	echo "Please add system configuration and try again. Exiting..."
	exit
fi
