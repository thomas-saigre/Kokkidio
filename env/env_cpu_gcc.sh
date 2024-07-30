
# white desktop @ 575
if [[ $(echo $USER) == "psneguir" ]]; then
	export EIGEN_ROOT="/home/psneguir/Projekte/eigen-3.4.0_normal_src/build"
    # export CUDACXX=/usr/local/cuda-12.3/bin/nvcc
	# /home/psneguir/Projekte/env/sycl_workspace/llvm/build
# hlrn-gpu
elif [[ $node_name == *"glogin"* ]] || [[ $node_name == *"ggpu"* ]]; then
	module cmake/3.26.4   gcc/9.3.0
	# compiler=$(command -v nvcc)
# TU HPC
elif [[ $node_name == "frontend"* ]] || [[ $node_name == "gpu"* ]]; then
	module load cmake/3.28.1
	# eigen="$(realpath "$(cat "$HOME/.cmake/packages/Eigen3/"*)"/..)"
elif [[ $node_name == "desktop-home-neon" ]]; then
	# /usr/local/cuda-12.4/bin is already in PATH, so no-op:
	:
	# optional:
	export EIGEN_ROOT="$HOME/pkg/eigen/build"
	export CC=`command -v gcc`
	export CXX=`command -v g++`
else
	echo "Please add system configuration and try again. Exiting..."
	exit
fi
