
# cmakeFlags+=" -DKOKKIDIO_USE_SYCL=ON"

setFlags() {
	if ! command -v clang++ ; then
		echo "clang++ not loaded, please run"
		echo "	source /path/to/oneAPI/setvars.sh --include-intel-llvm"
		echo "and retry."
		exit
	fi
	export SYCL_TARGET=nvidia
	export SYCL_ROOT=$(realpath $(dirname $(command -v clang++))/../..)
	cmakeFlags+=" -DCMAKE_C_COMPILER=$(command -v clang)"
	cmakeFlags+=" -DCMAKE_CXX_COMPILER=$(command -v clang++)"
}

if [[ $(echo $USER) == "psneguir" ]]; then
	export EIGEN_ROOT="/home/psneguir/Projekte/eigen-3.4.0_normal_src/build"
	export SYCL_ROOT="/home/psneguir/Projekte/env/sycl_workspace/llvm/build"
	cmakeFlags+=" -DCMAKE_C_COMPILER=${SYCL_ROOT}/bin/clang"
	cmakeFlags+=" -DCMAKE_CXX_COMPILER=${SYCL_ROOT}/bin/clang++"
	## for setting the GPU target manually via cmake var
	# cmakeFlags+=" -DSYCL_TARGET=nvidia"
	## for setting the GPU target manually via env var:
	# export SYCL_TARGET=nvidia,amd
	# export SYCL_TARGET=nvidia
	# export SYCL_TARGET=bad
# HLRN, grete
elif [[ $node_name == *"glogin"* ]] || [[ $node_name == *"ggpu"* ]]; then
	# This is required by libdevice, otherwise we need to pass --cuda-path or -nocudalib
	module load cuda/12.1   cmake/3.26.4
	# source /scratch-emmy/usr/`whoami`/oneAPI/setvars.sh --include-intel-llvm
	setFlags
	export EIGEN_ROOT="/scratch-emmy/usr/`whoami`/eigen/build"
# HLRN, Intel PVC
elif [[ $node_name == "bgi"* ]]; then
	module load intel
	setFlags
	export SYCL_TARGET=intel
	export EIGEN_ROOT="$HOME/pkg/eigen/build"
# TU HPC
elif [[ $node_name == "frontend"* ]] || [[ $node_name == "gpu"* ]]; then
	module load cmake/3.28.1
	# This is required by libdevice, otherwise we need to pass --cuda-path or -nocudalib
	module load nvidia/cuda/12.2
	# source /scratch/wahyd/env/oneAPI/setvars.sh --include-intel-llvm
	setFlags
	export EIGEN_ROOT="/scratch/wahyd/eigen/build"
# lenni desktop
elif [[ $node_name == "desktop-home-neon" ]]; then
	# export EIGEN_ROOT="$HOME/pkg/eigen/build"
	ONEAPI_BASE="/opt/intel/oneapi"
	export oneDPL_ROOT="$ONEAPI_BASE/dpl/latest/lib/cmake/"
	ONEAPI_INST="$ONEAPI_BASE/compiler/2024.1"
	export PATH=$(prepend_path "$ONEAPI_INST/bin" "${PATH:-}")
	export LD_LIBRARY_PATH=$(prepend_path "${ONEAPI_INST}/lib" "${LD_LIBRARY_PATH:-}")
	export SYCL_ROOT="$ONEAPI_INST"

	Kokkos_ARCH=Kokkos_ARCH_VOLTA70

	comp="intel"
	# comp="clang"
	if [[ $comp == "intel" ]]; then
		export CC="$ONEAPI_INST/bin/icx"
		export CXX="$ONEAPI_INST/bin/icpx"
		export OMP_LIB_DIR="$ONEAPI_INST/lib"
	elif [[ $comp == "clang" ]]; then
		# setFlags
		ONEAPI_LLVM="$ONEAPI_INST/bin/compiler"
		export OMP_LIB_DIR="$ONEAPI_INST/lib"
		export CC="$ONEAPI_LLVM/clang"
		export CXX="$ONEAPI_LLVM/clang"
		export PATH=$(prepend_path "$ONEAPI_LLVM" "${PATH:-}")
		# export LD_LIBRARY_PATH=$(prepend_path "${ONEAPI_INST}/lib" "${LD_LIBRARY_PATH:-}")
	fi
else
	echo "Please add system configuration and try again. Exiting..."
	exit
fi
