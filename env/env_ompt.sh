
# cmakeFlags+=" -DKOKKIDIO_USE_OMPT=ON"

check_oneapi () {
	if ! command -v icpx ; then
		echo "icpx not loaded, please run"
		echo "	source /path/to/oneAPI/setvars.sh --include-intel-llvm"
		echo "and retry."
		exit
	fi
}

# white desktop @ 575
if [[ $(echo $USER) == "psneguir" ]]; then
	export EIGEN_ROOT="/home/psneguir/Projekte/eigen-3.4.0_normal_src/build"
    export CUDACXX=/usr/local/cuda-12.3/bin/nvcc
	# /home/psneguir/Projekte/env/sycl_workspace/llvm/build
# hlrn-gpu
elif [[ $node_name == *"glogin"* ]] || [[ $node_name == *"ggpu"* ]]; then
	PKGDIR="/scratch-emmy/usr/`whoami`"
	module load cuda/12.1   cmake/3.26.4
	# optional:
	# export EIGEN_ROOT="$PKGDIR/eigen/build"
	# compiler=$(command -v nvcc)

	# comp="intel"
	comp="clang"

	if [[ $comp == "intel" ]]; then
		check_oneapi
		cmakeFlags+=" -DCMAKE_C_COMPILER=$(command -v icx)"
		cmakeFlags+=" -DCMAKE_CXX_COMPILER=$(command -v icpx)"
		export OMP_TARGET=nvidia
	elif [[ $comp == "clang" ]]; then
		LLVM_INST="$PKGDIR/llvm"
		export CC="$LLVM_INST/bin/clang"
		export CXX="$LLVM_INST/bin/clang++"
		# cmakeFlags+=" -DCMAKE_C_COMPILER=$LLVM_INST/bin/clang"
		# cmakeFlags+=" -DCMAKE_CXX_COMPILER=$LLVM_INST/bin/clang++"
		# export OMP_LIB_DIR="$LLVM_INST/lib/x86_64-unknown-linux-gnu"
		# cmakeFlags+=" -DCMAKE_MODULE_PATH=$OMP_LIB_DIR/cmake/openmp"
	fi
# TU HPC
# HLRN, Intel PVC
elif [[ $node_name == "bgi"* ]]; then
	module load intel
	cmakeFlags+=" -DCMAKE_C_COMPILER=$(command -v icx)"
	cmakeFlags+=" -DCMAKE_CXX_COMPILER=$(command -v icpx)"
	export OMP_TARGET=intel
	export EIGEN_ROOT="$HOME/pkg/eigen/build"
# TU HPC
elif [[ $node_name == "frontend"* ]] || [[ $node_name == "gpu"* ]]; then
	module load cmake/3.28.1
	module load nvidia/cuda/12.2
	# eigen="$(realpath "$(cat "$HOME/.cmake/packages/Eigen3/"*)"/..)"
# lenni desktop
elif [[ $node_name == "desktop-home-neon" ]]; then
	# optional:
	# export EIGEN_ROOT="$HOME/pkg/eigen/build"

	Kokkos_ARCH=Kokkos_ARCH_VOLTA70

	# comp="intel"
	comp="clang"
	# comp="gnucc"

	if [[ $comp == "intel" ]]; then
		# IntelLLVM:
		# check_oneapi
		# cmakeFlags+=" -DCMAKE_C_COMPILER=$(command -v icx)"
		# cmakeFlags+=" -DCMAKE_CXX_COMPILER=$(command -v icpx)"
		# export OMP_TARGET=nvidia
		ONEAPI_BASE="/opt/intel/oneapi"
		ONEAPI_INST="$ONEAPI_BASE/compiler/2024.1"
		export CC="$ONEAPI_INST/bin/icx"
		export CXX="$ONEAPI_INST/bin/icpx"
		export OMP_LIB_DIR="$ONEAPI_INST/lib"
	elif [[ $comp == "clang" ]]; then
		# clang:
		LLVM_INST="$HOME/pkg/llvm/install"
		export CC="$LLVM_INST/bin/clang"
		export CXX="$LLVM_INST/bin/clang++"
		export OMP_LIB_DIR="$LLVM_INST/lib/x86_64-unknown-linux-gnu"
		# cmakeFlags+=" -DCMAKE_MODULE_PATH=$OMP_LIB_DIR/cmake/openmp"
	elif [[ $comp == "gnucc" ]]; then
		# GNU CC
		GCC_INST="$HOME/pkg/omp-gcc/gcc-13.2.0"
		export CC="$GCC_INST/bin/gcc"
		export CXX="$GCC_INST/bin/g++"
		export OMP_LIB_DIR="$GCC_INST/lib64"
		# export PATH="$GCC_INST/libexec/gcc/x86_64-pc-linux-gnu/13.2.0/accel/nvptx-none/:$PATH"
# 		export PATH="$GCC_INST/bin:$PATH"
		# export OMP_ARCH=nvptx-none
	fi
else
	echo "Please add system configuration and try again. Exiting..."
	exit
fi
