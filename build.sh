#!/bin/bash

set -eu

# script path
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# used to determine which file in ./env/ to read machine-specific variables from
node_name=$(uname -n)
env_dir="${sd}/env"

source "${sd}/scripts/build_parseOpts.sh"
source "${sd}/scripts/build_func.sh"

read_nodefile
download_libs

# Eigen is independent of the build type (Debug/Release)
if [[ $buildEigen == true ]]; then
	build_eigen
fi


build_backend () {
	set_backend
	for buildtype in "${buildtypes[@]}"; do
		if [[ "$backend_default" == "cuda" ]] && [[ "$buildtype" == "Debug" ]]; then
			if [[ $backend =~ ompt|sycl ]]; then
				printf '%s%s%s\n' \
					"Backend=${backend^^} on CUDA machine -> " \
					"skipping build type \"Debug\" " \
					"due to ptxas error in Debug builds with ${backend^^}"
				continue
			fi
		fi
		# set_vars $buildtype
		if [[ $buildKokkos == true ]]; then
			build_kokkos $buildtype
		fi
		if [[ $buildKokkidio == true ]]; then
			build_kokkidio $buildtype
		fi
		if [[ $buildTests == true ]]; then
			check_kokkos_install true
			for sc in float double; do
				if ! [[ $whichScalar =~ all|$sc ]]; then
					continue
				fi
				build_tests $buildtype $sc
			done
		fi
		if [[ $buildExamples == true ]]; then
			check_kokkos_install true
			build_examples $buildtype
		fi
	done
	echo "Finished compilation(s) for ${backend^^}."
}


if [[ "$backend" == "all" ]]; then
	for b in cuda hip ompt sycl cpu_gcc; do
		backend="$b"
		build_backend
	done
else
	build_backend
fi


