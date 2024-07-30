#!/bin/sh

set -euv

# following this recipe
# https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm
# and including some parts of this:
# https://hpc-wiki.info/hpc/Building_LLVM/Clang_with_OpenMP_Offloading_to_NVIDIA_GPUs

nproc=32

# set CUDA architectures (sm_xx):
SM_XX=35,60,70,75,80
# set CUDA default architecture
SM_DEF=sm_60

isLocal=true
isLocal=false

if [[ $isLocal == true ]]; then
	WORKDIR=$HOME/pkg/llvm
	INSTDIR=$WORKDIR/install
	nproc=32
else
	WORKDIR=/dev/shm/__`whoami`
	INSTDIR=/scratch-emmy/usr/`whoami`/llvm
	nproc=20
	# I think using a somewhat modern GCC is optional, but could be helpful
	# cuda is requried
	module load cuda/12.1 gcc/9.3.0
fi

LLVM_DIR=$WORKDIR/llvm-project
LLVM_BUILD=$LLVM_DIR/build

# shallow clone
git clone --depth 1 https://github.com/llvm/llvm-project.git "$LLVM_DIR"
cd "$LLVM_DIR"

# configure (max 4 link jobs, because I ran out of memory on a 64GB machine)
cmake -S llvm -B build -G "Unix Makefiles" \
	-DLLVM_ENABLE_PROJECTS='clang;openmp' \
	-DCMAKE_INSTALL_PREFIX=$INSTDIR \
	-DCMAKE_BUILD_TYPE=Release \
	-DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=$SM_DEF \
	-DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=$SM_XX \
	-DLLVM_PARALLEL_LINK_JOBS=4

cmake --build $LLVM_BUILD -j${nproc}
cmake --build $LLVM_BUILD -j${nproc} --target install
