
# cmakeFlags+=" -DKOKKIDIO_USE_CUDA=ON"

# white desktop @ 575
if [[ $(echo $USER) == "psneguir" ]]; then
    export CUDACXX=/usr/local/cuda-12.3/bin/nvcc
	# /home/psneguir/Projekte/env/sycl_workspace/llvm/build

# hlrn-gpu
elif [[ $node_name == *"glogin"* ]] || [[ $node_name == *"ggpu"* ]]; then
	module load cuda/12.1   gcc/9.3.0
	# compiler=$(command -v nvcc)

# TU HPC
elif [[ $node_name == "frontend"* ]] || [[ $node_name == "gpu"* ]]; then
	module load nvidia/cuda/12.2
	# eigen="$(realpath "$(cat "$HOME/.cmake/packages/Eigen3/"*)"/..)"

# lenni home office
elif [[ $node_name == "desktop-home-neon" ]]; then
	Kokkos_ARCH=Kokkos_ARCH_VOLTA70
	# cmakeFlags+=" -DKokkos_ARCH_VOLTA70=ON"
	# gcc_path=`command -v g++`
	# cmakeFlags+=" -DCMAKE_CXX_COMPILER=${gcc_path}"
else
	echo "Please add system configuration and try again. Exiting..."
	exit
fi
