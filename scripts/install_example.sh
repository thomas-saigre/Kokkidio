#!/bin/bash

# Kokkos and Eigen must already be installed.

# script path
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# proj root
kokkidio_root="${sd}/.."

# build dir
kokkidio_build="${kokkidio_root}/_example/build"

# install dir
kokkidio_inst="${kokkidio_root}/_example/install"

# backend
# backend=cuda
backend=ompt
# backend=sycl
# backend=cpu_gcc

if [[ "$backend" == "ompt" ]]; then
	export OMP_LIB_DIR="$HOME/pkg/llvm/install/lib/x86_64-unknown-linux-gnu"
elif [[ "$backend" == "sycl" ]]; then
	export OMP_LIB_DIR="/opt/intel/oneapi/compiler/2024.1/lib"
fi

cmake \
	-S "${kokkidio_root}" \
	-B "${kokkidio_build}" \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_EXTENSIONS=Off \
	-D EIGEN_ROOT=$HOME/pkg/eigen \
	-D Kokkos_ROOT=$HOME/pkg/kokkos/install/${backend}/Release \

cmake --install \
	"${kokkidio_build}" \
	--prefix ${kokkidio_inst}

cmake \
	-S "${kokkidio_root}/src/bench" \
	-B "${kokkidio_build}/bench" \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_EXTENSIONS=Off \
	-D Kokkidio_ROOT="${kokkidio_inst}"