#!/bin/bash

set -eu

# script path
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# used to determine which file in ./env/ to read machine-specific variables from
node_name=$(uname -n)
env_dir="${sd}/env"

source "${sd}/scripts/build_parseOpts.sh"


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
	local buildtype=$1

	# reset cmakeFlags between runs/backends etc.
	cmakeFlags=""
	cmakeFlags+=" -DCMAKE_BUILD_TYPE=$buildtype"
	# cmakeFlags+=" -DCMAKE_CXX_STANDARD=17"
	# Kokkos otherwise emits a warning
	cmakeFlags+=" -DCMAKE_CXX_EXTENSIONS=OFF"

	echo "Retrieving system-specific variables from \"${env_dir}\" ..."

	source "$env_dir/node_patterns.sh"
	nodefile="$env_dir/nodes/${node_name}.sh"
	source "$nodefile"
	# source "$env_dir/env_tpl.sh"
	if [[ "$backend" == "default" ]]; then
		if [ -n "${backend_default+x}" ]; then
			backend=$backend_default
		else
			printf '%s \n%s %s' \
				"Backend not specified and backend_default not defined." \
				"Please either specify backend as command line option," \
				"or set the variable \"backend_default\" in $nodefile."
			print_help
			exit
		fi
	fi
	echo "Using backend: $backend"

	if [ ! -n "${Kokkos_SRC+x}" ]; then
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
				if [ -n "${Kokkos_INST+x}" ]; then
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
}

set_gpu_arch() {
	local required=${1:-false}
	if [ -n "${Kokkos_ARCH+x}" ]; then
		cmakeFlags+=" -D${Kokkos_ARCH}=ON"
		# Kokkos requires this for SYCL
		if [[ "$backend" == "sycl" ]]; then
			cmakeFlags+=" -DKokkos_ENABLE_UNSUPPORTED_ARCHS=ON"
		fi
	else
		local arch_help=$(printf '%s\n%s\n%s\n' \
			"Kokkos_ARCH=Kokkos_ARCH_<specifier>" \
			"You can find the specifiers here:" \
			"https://kokkos.org/kokkos-core-wiki/keywords.html#architectures" \
		)
		if [[ $required == true ]]; then
			echo "Please specify the GPU architecture in $nodefile as"
			echo "$arch_help"
			exit
		else
			echo "No GPU architecture specified, falling back to autodetection."
			echo "You may explicitly specify the GPU architecture in $nodefile using"
			echo "$arch_help"
		fi
	fi
}

set_kokkos_targets () {
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
	local buildtype=$1

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
	local kk_testfile_tail="cmake/Kokkos/KokkosConfig.cmake"
	for subdir in lib lib64; do
		local kk_testfile="$Kokkos_ROOT/$subdir/$kk_testfile_tail"
		echo "Checking for Kokkos cmake config \"$kk_testfile\"..."
		if [ -f "$kk_testfile" ]; then
			echo "Kokkos_ROOT dir: $Kokkos_ROOT"
			return
		fi
	done

	echo "Could not find Kokkos cmake config file! Did you install after building?"
	exit
}

build_kokkidio () {
	local buildtype=$1

	make_title "Configuring Kokkidio for ${backend^^}, build type \"$buildtype\"."

	builddir="_build/kokkidio/$backend/$buildtype"
	set_kokkos_targets "$backend"
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
			check_kokkos_build $buildtype
			build_kokkidio $buildtype
		fi
		if [[ $buildTests == true ]]; then
			check_kokkos_build $buildtype
			for sc in float double; do
				if ! [[ $whichScalar =~ all|$sc ]]; then
					continue
				fi
				build_tests $buildtype $sc
			done
		fi
		if [[ $buildExamples == true ]]; then
			check_kokkos_build $buildtype
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
