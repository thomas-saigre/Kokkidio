#!/bin/bash

set -eu

# script path
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source "${sd}/scripts/parseOpts.sh"


# used by env/env_<backend>.sh
node_name=$(uname -n)

# copied from oneapi
prepend_path() (
  path_to_add="$1"
  path_is_now="$2"

  if [ "" = "${path_is_now}" ] ; then   # avoid dangling ":"
    printf "%s" "${path_to_add}"
  else
    printf "%s" "${path_to_add}:${path_is_now}"
  fi
)

make_title () {
	echo "----------------"
	echo "$1"
	echo "----------------"
}

# cmakeFlags=""

set_vars () {
	local backend=$1
	local buildtype=$2

	# reset cmakeFlags between runs/backends etc.
	cmakeFlags=""
	cmakeFlags+=" -DCMAKE_BUILD_TYPE=$buildtype"
	# cmakeFlags+=" -DCMAKE_CXX_STANDARD=17"
	# Kokkos otherwise emits a warning
	cmakeFlags+=" -DCMAKE_CXX_EXTENSIONS=OFF"

	echo "Retrieving third party library (TPL) variables..."
	source "${sd}/env/env_tpl.sh"

	if [ ! -v Kokkos_SRC ]; then
		echo "Kokkos source directory not specified."
		printf '%s%s\n' \
			"Please set the environment variable \"Kokkos_SRC\" " \
			"to the Kokkos source directory!"
		exit
	fi

	local kk_testfile="$Kokkos_SRC/cmake/KokkosConfig.cmake.in"

	if [ -f "$kk_testfile" ]; then
		echo "Kokkos source dir: $Kokkos_SRC"
	else
		echo "Could not find test file: $kk_testfile"
		exit
	fi

	if [[ $install_prefix != "" ]]; then
		echo $install_prefix
		if [[ $buildKokkos == true ]]; then
			if [[ $buildKokkidio != true ]]; then
				if [ -v Kokkos_INST ]; then
					printf '%s \n%s \n%s \n%s \n%s \n' \
						"Found both environment variable " \
						"	Kokkos_INST=$Kokkos_INST" \
						"and command line option " \
						"	--prefix=$install_prefix." \
						"Command line option takes precedence."
				fi
				Kokkos_INST="$install_prefix"
			else
				echo "Using --prefix=$install_prefix as install prefix for Kokkidio."
				Kokkidio_INST="$install_prefix"
			fi
		fi
	else
		Kokkidio_INST=""
	fi

	Kokkidio_ROOT="${Kokkidio_INST:-$sd/_install}/${backend}/$buildtype"
	Kokkos_BUILD="${Kokkos_BUILD:-$Kokkos_SRC/_build}/${backend}/$buildtype"
	Kokkos_ROOT="${Kokkos_INST:-$Kokkos_SRC/_install}/${backend}/$buildtype"

	echo "Retrieving environment variables..."
	KOKKIDIO_SKIP_BACKEND=false
	source "$sd/env/env_${backend}.sh"
}

set_gpu_arch() {
	local backend=$1
	local required=${2:-false}
	if [ -v Kokkos_ARCH ]; then
		cmakeFlags+=" -D${Kokkos_ARCH}=ON"
		# Kokkos requires this for SYCL
		if [[ "$backend" == "sycl" ]]; then
			cmakeFlags+=" -DKokkos_ENABLE_UNSUPPORTED_ARCHS=ON"
		fi
	else
		if [[ $required == true ]]; then
			printf '%s\n\t%s\n%s%s' \
				"Please specify the GPU architecture in env_$backend.sh as" \
				"Kokkos_ARCH=Kokkos_ARCH_<specifier>" \
				"You can find the specifiers here: " \
				"https://kokkos.org/kokkos-core-wiki/keywords.html#architectures"
			exit
		else
			echo "No GPU architecture specified, falling back to autodetection."
			echo "You may explicitly specify the GPU architecture in env_$backend.sh."
		fi
	fi
}

set_kokkos_targets () {
	local backend=$1

	# translate to Kokkos naming
	if [[ $backend == "ompt" ]]; then
		local kokkos_backend="openmptarget"
	else
		local kokkos_backend="$backend"
	fi

	if [[ ! $backend =~ cpu ]]; then
		cmakeFlags+=" -DKokkos_ENABLE_${kokkos_backend^^}=ON"
	fi
}

build_cmake () {
	local builddir="$1"
	local srcdir="$2"
	local instdir="${3:-""}"
	echo "Build directory: $builddir"
	echo "Source directory: $srcdir"
	if [ -n "$instdir" ]; then
		echo "Install directory: $instdir"
		cmakeFlags+=" -DCMAKE_INSTALL_PREFIX=$instdir"
	fi
	echo "CMake flags: $cmakeFlags"
	mkdir -p "$builddir"
	cmake -B "$builddir" $cmakeFlags ${srcdir}

	if [[ $noBuild == true ]]; then
		echo "Build with \"cmake --build $builddir -j\""
	else
		echo "Building with CMake..."
		cmake --build "$builddir" -j
		if [[ $install_opt == true ]] && [[ $instdir != "" ]]; then
			cmake --build "$builddir" -- install
		fi
	fi
}

build_kokkos () {
	local backend=$1
	local buildtype=$2

	make_title "Building Kokkos for ${backend^^}, build type \"$buildtype\"."

	set_kokkos_targets "$backend"
	cmakeFlags+=" -DKokkos_ENABLE_OPENMP=ON"
	cmakeFlags+=" -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON"
	if [[ "$backend" == "cuda" ]]; then
		set_gpu_arch $backend true
		cmakeFlags+=" -DKokkos_ENABLE_CUDA_CONSTEXPR=ON"
		cmakeFlags+=" -DKokkos_ENABLE_CUDA_LAMBDA=ON"
	fi
	if [[ $backend =~ sycl|ompt ]]; then
		set_gpu_arch $backend
	fi

	build_cmake "$Kokkos_BUILD" "$Kokkos_SRC" "$Kokkos_ROOT"
}

check_kokkos_build () {
	echo "Checking Kokkos build..."
	local kk_testfile="$Kokkos_ROOT/lib/cmake/Kokkos/KokkosConfig.cmake"
	if [ -f "$kk_testfile" ]; then
		echo "Kokkos_ROOT dir: $Kokkos_ROOT"
	else
		echo "Could not find test file: $kk_testfile"
		exit
	fi
}

build_kokkidio () {
	local backend=$1
	local buildtype=$2

	make_title "Configuring Kokkidio for ${backend^^}, build type \"$buildtype\"."

	builddir="_build/kokkidio/$backend/$buildtype"
	set_kokkos_targets "$backend"
	cmakeFlags+=" -DKokkos_ROOT=$Kokkos_ROOT"

	echo "Running build commands..."
	build_cmake "$builddir" "$sd" "$Kokkidio_ROOT"
}

build_tests () {
	local backend=$1
	local buildtype=$2
	local scalar=$3

	make_title "Building tests with ${backend^^}, build type \"$buildtype\", scalar type \"$scalar\"."

	export KOKKIDIO_REAL_SCALAR=$scalar
	builddir="_build/tests/$backend/$buildtype/$scalar"

	set_kokkos_targets "$backend"
	cmakeFlags+=" -DKokkos_ROOT=$Kokkos_ROOT"
	cmakeFlags+=" -DKokkidio_ROOT=$Kokkidio_ROOT"

	echo "Running build commands..."
	build_cmake "$builddir" "$sd/tests"


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

build_all () {
	local backend=$1
	if [[ "$whichBackend" =~ all|$backend ]]; then

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
			set_vars $backend $buildtype
			if [[ $KOKKIDIO_SKIP_BACKEND == true ]]; then
				continue
			fi
			if [[ $buildKokkos == true ]]; then
				build_kokkos $backend $buildtype
			fi
			if [[ $buildKokkidio == true ]]; then
				check_kokkos_build $backend $buildtype
				build_kokkidio $backend $buildtype
			fi
			if [[ $buildTests == true ]]; then
				check_kokkos_build $backend $buildtype
				for sc in float double; do
					if ! [[ $whichScalar =~ all|$sc ]]; then
						continue
					fi
					build_tests $backend $buildtype $sc
				done
			fi
		done
		echo "Finished compilation(s) for ${backend^^}."
	else
		echo "Skipping compilation(s) for ${backend^^}."
	fi
}

for b in cuda hip ompt sycl cpu_gcc; do
	build_all $b
done
