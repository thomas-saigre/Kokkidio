
if [[ $node_name == *"login"* ]] || [[ $node_name == *"ggpu"* ]]; then
	module load cuda/12.1
	module load gcc/9.3.0

	export ENV_PATH=/scratch-emmy/usr/`whoami`
	export EIGEN_ROOT=${ENV_PATH}/eigen
	export ROCM_ROOT=${ENV_PATH}/rocm/5.7.x

	# export HIP_PATH=${ROCM_ROOT}
	export HIP_PLATFORM=nvidia
	# export HIP_COMPILER=nvcc
	# export HIP_RUNTIME=cuda
	# export CUDA_PATH=/usr/lib/nvidia-cuda-toolkit
	# export PATH=${HIP_PATH}/bin:${PATH}
	export PATH=${ROCM_ROOT}/bin:${PATH}

	# let's also suppress an annoying perl locale warning
	# export LC_CTYPE=en_GB.UTF-8
	export LC_ALL=en_GB.UTF-8
	export LC_MEASUREMENT=en_DE.UTF-8
elif [[ $node_name == "desktop-home-neon" ]]; then
	KOKKIDIO_SKIP_BACKEND=true
	# export ROCM_ROOT=/opt/rocm
	# export EIGEN_ROOT="$HOME/pkg/eigen"
	# export HIP_PLATFORM=nvidia
	# cmakeFlags+=" -DKokkos_ARCH_VOLTA70=ON"
fi
