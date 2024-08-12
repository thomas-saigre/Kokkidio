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
# exit

# check_install_prefix () {

# 	if [ -n $install_prefix ]; then
# 		echo "Install prefix: \"$install_prefix\""
# 		if [[ $buildLib == true ]]; then
# 			if [[ $buildKokkidio != true ]]; then
# 				if [ -n "${Kokkos_INST+x}" ]; then
# 					printf '%s \n%s \n%s \n%s \n%s \n' \
# 						"Found both environment variable " \
# 						"	Kokkos_INST=$Kokkos_INST" \
# 						"and command line option " \
# 						"	--prefix=$install_prefix." \
# 						"Command line option takes precedence."
# 				fi
# 				Kokkos_INST="$install_prefix"
# 			else
# 				echo "Using --prefix=$install_prefix as install prefix for Kokkidio."
# 				Kokkidio_INST="$install_prefix"
# 			fi
# 		fi
# 	else
# 		Kokkidio_INST=""
# 	fi
# }

set_vars () {
	local buildtype=$1

	reset_cmake_flags
	set_backend

	check_kokkos_src

	# check_install_prefix

	Kokkidio_ROOT="${Kokkidio_INST:-$sd/_install}/${backend}/$buildtype"
	# Kokkos_BUILD="${Kokkos_BUILD:-$Kokkos_SRC/_build}/${backend}/$buildtype"
	# Kokkos_ROOT="${Kokkos_INST:-$Kokkos_SRC/_install}/${backend}/$buildtype"
}

build_kokkos () {
	local buildtype=$1

	make_title "Building Kokkos for ${backend^^}, build type \"$buildtype\"."

	check_kokkos_src
	set_kokkos_targets $backend
	set_kokkos_root $buildtype
	Kokkos_BUILD="${Kokkos_BUILD:-$Kokkos_SRC/_build}/${backend}/$buildtype"

	reset_cmake_flags

	cmakeFlags+=" -DKokkos_ENABLE_OPENMP=ON"
	cmakeFlags+=" -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON"
	if [[ $backend =~ cuda|hip ]]; then
		set_gpu_arch $backend true
		cmakeFlags+=" -DKokkos_ENABLE_CUDA_CONSTEXPR=ON"
		cmakeFlags+=" -DKokkos_ENABLE_CUDA_LAMBDA=ON"
	fi
	if [[ $backend =~ sycl|ompt ]]; then
		set_gpu_arch $backend
	fi

	build_cmake "$Kokkos_BUILD" "$Kokkos_SRC" "$Kokkos_ROOT"
}

build_eigen () {
	make_title "Building Eigen."

	check_eigen_src
	set_eigen_root
	reset_cmake_flags

	build_cmake "$Eigen_ROOT" "$Eigen_SRC"
}

build_kokkidio () {
	local buildtype=$1

	make_title "Configuring Kokkidio for ${backend^^}, build type \"$buildtype\"."

	check_kokkos_install $buildtype
	set_kokkos_targets $backend
	set_kokkos_root $buildtype

	builddir="_build/kokkidio/$backend/$buildtype"
	cmakeFlags+=" -DKokkos_ROOT=$Kokkos_ROOT"

	echo "Running build commands..."
	build_cmake "$builddir" "$sd" "$Kokkidio_ROOT"
}

build_tests () {
	local buildtype=$1
	local scalar=$2

	make_title "Building tests with ${backend^^}, build type \"$buildtype\", scalar type \"$scalar\"."

	export KOKKIDIO_REAL_SCALAR=$scalar
	builddir="_build/tests/$backend/$buildtype/$scalar"

	set_kokkos_targets
	cmakeFlags+=" -DKokkos_ROOT=$Kokkos_ROOT"
	cmakeFlags+=" -DKokkidio_ROOT=$Kokkidio_ROOT"

	echo "Running build commands..."
	build_cmake "$builddir" "$sd/src/tests"


	# if [[ $backend != "hip" ]]; then
	# 	cmakeFlags+=" -DCMAKE_INSTALL_PREFIX=${sd}/install"
	# 	build_cmake "$builddir" "$sd" true
	# 	build_cmake "$builddir/tests" "$sd/tests"
	# elif [[ $noBuild == true ]]; then
	# 	export KOKKIDIO_BUILDTYPE=$buildtype
	# 	make -j
	# fi

	echo "Finished compilation for ${backend^^}, build type \"$buildtype\", scalar type \"$scalar\"."

	if [[ $backend =~ ompt|sycl ]]; then
		runbuild=Release
	else
		runbuild=Debug
	fi

	if [[ $noRun != true ]] && [[ $buildtype == $runbuild ]]; then
		echo "Performing test runs in $buildtype mode..."
		# Without these OMP settings, Kokkos complains about their absence.
		export OMP_PROC_BIND=spread 
		export OMP_PLACES=threads
		$builddir/dotProduct/dotProduct --size 4 50 --runs 2
		$builddir/friction/friction --size 50 --runs 2
		echo "Test runs in $buildtype mode finished."
	fi
}

build_examples () {
	local buildtype=$1

	make_title "Building examples with ${backend^^}, build type \"$buildtype\"."

	export KOKKIDIO_REAL_SCALAR=float
	builddir="_build/examples/$buildtype"

	set_kokkos_targets
	cmakeFlags+=" -DKokkos_ROOT=$Kokkos_ROOT"
	cmakeFlags+=" -DKokkidio_ROOT=$Kokkidio_ROOT"

	echo "Running build commands..."
	build_cmake "$builddir" "$sd/src/examples"

	echo "Finished compilation for ${backend^^}, build type \"$buildtype\"."
	printf 'Executables can be found in \n\t%s\n' \
		"${builddir}"
}

build_backend () {
	for buildtype in "${buildtypes[@]}"; do
		if [[ $buildtype == "Debug" ]]; then
			if [[ $backend =~ ompt|sycl ]]; then
				printf '%s%s%s\n' \
					"Backend=${backend^^} -> " \
					"skipping build type \"Debug\" " \
					"due to ptxas error in Debug builds with ${backend^^}"
				continue
			fi
		fi
		set_vars $buildtype
		if [[ $buildKokkos == true ]]; then
			build_kokkos $buildtype
		fi
		if [[ $buildKokkidio == true ]]; then
			build_kokkidio $buildtype
		fi
		if [[ $buildTests == true ]]; then
			check_kokkos_install $buildtype
			for sc in float double; do
				if ! [[ $whichScalar =~ all|$sc ]]; then
					continue
				fi
				build_tests $buildtype $sc
			done
		fi
		if [[ $buildExamples == true ]]; then
			check_kokkos_install $buildtype
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


