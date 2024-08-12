


download_lib () {
	local subj="$1"
	local url="$2"
	# local srcdir="$3"
	# if [[ "$srcdir" != "x" ]]; then
	declare srcdir="${subj}_SRC"
	declare rootdir="${subj}_ROOT"

	if [ -n "${!rootdir+x}" ]; then
		if [ -d "${!rootdir}" ]; then
			printf '%s\n%s\n%s\n' \
				"Directory" \
				"    ${rootdir}=\"${!rootdir}\"" \
				"found. Skipping download..."
			return
		else
			printf '%s\n%s\n%s\n' \
				"Variable" \
				"    ${rootdir}=\"${!rootdir}\"" \
				"found, but is not a directory!"
			printf '%s\n%s\n' \
				"Please define source directory via variable \"${srcdir}\"." \
				"Exiting..."
			exit
		fi
	fi

	if [ -n "${!srcdir+x}" ]; then
		echo "yay! running: git clone $url ${!srcdir}"
		exit
		git clone "$url" "${srcdir}"
	else
		echo "${subj}_SRC not set! Please set it in your node file, e.g."
		echo "${nodefile}"
		exit
	fi
}


download_libs () {
	if [[ $download_opt != true ]]; then
		return
	fi
	if [[ $buildKokkos == true ]]; then
		download_lib Kokkos https://github.com/kokkos/kokkos.git
	fi
	if [[ $buildEigen == true ]]; then
		download_lib Eigen https://gitlab.com/libeigen/eigen.git
	fi
}


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

read_nodefile () {
	echo "Retrieving system-specific variables from \"${env_dir}\" ..."

	source "$env_dir/node_patterns.sh"
	nodefile="$env_dir/nodes/${node_name}.sh"
	if [ -f "$nodefile" ]; then
		echo "Reading node file \"$nodefile\""
		source "$nodefile"
	else
		printf '%s \n%s \n%s %s \n%s' \
			"Could not find node file to set machine-specific variables." \
			"Creating default node file: " "$nodefile" \
			"In that file, you must at least specify the default backend," \
			"and the target architecture." \
			"Rerun this script afterwards. Exiting..."
		cp "$sd/scripts/nodefile_base.sh" "$nodefile"
	fi
}

set_backend () {
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
}

check_src () {
	local subj=$1
	local testfile_suffix="$2"
	declare srcdir="${subj}_SRC"
	if [ -z "${!srcdir:+x}" ]; then
		echo "$subj source directory not specified."
		printf '%s%s\n' \
			"Please set the environment variable \"$srcdir\" " \
			"to the $subj source directory!"
		exit
	fi

	local testfile="${!srcdir}/$testfile_suffix"

	if [ -f "$testfile" ]; then
		echo "$subj source dir: ${!srcdir}"
	else
		echo "Could not find test file: $testfile"
		exit
	fi
}

check_kokkos_src () {
	check_src "Kokkos" "cmake/KokkosConfig.cmake.in"
	# if [ -z "${Kokkos_SRC+x}" ]; then
	# 	echo "Kokkos source directory not specified."
	# 	printf '%s%s\n' \
	# 		"Please set the environment variable \"Kokkos_SRC\" " \
	# 		"to the Kokkos source directory!"
	# 	exit
	# fi

	# local testfile="$Kokkos_SRC/cmake/KokkosConfig.cmake.in"

	# if [ -f "$testfile" ]; then
	# 	echo "Kokkos source dir: $Kokkos_SRC"
	# else
	# 	echo "Could not find test file: $kk_testfile"
	# 	exit
	# fi
}

check_eigen_src () {
	check_src "Eigen" "cmake/Eigen3Config.cmake.in"
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

# set install prefix according to command line and script vars
set_root () {

	declare varname="$1"
	local defaultval="$2"
	local instdir="${!varname:-$defaultval}"

	if [[ $buildSingle == true ]] && [ -n $install_prefix ]; then
		instdir="$install_prefix"
		if [ -n "${!varname+x}" ]; then
			echo "$varname=${!varname}" >&2
			printf '%s \n%s \n%s \n%s \n%s \n' \
				"Both variable " \
				"    ${varname}=\"${!varname}\"" \
				"and command line option " \
				"    --prefix=\"${install_prefix}\"" \
				"found. Command line option takes precedence." \
				>&2
		fi
	fi
	echo "$instdir"
}

set_kokkos_root () {
	local buildtype=$1
	Kokkos_ROOT="$(set_root Kokkos_INST "$Kokkos_SRC/_install")"
	Kokkos_ROOT+="/$backend/$buildtype"
}

set_eigen_root () {
	if [ -n "${Eigen_ROOT+x}" ] || [ -n "${Eigen_SRC+x}" ]; then
		Eigen_ROOT="${Eigen_ROOT:-$Eigen_SRC/build}"
	else
		printf '%s\n%s\n%s\n%s\n' \
			"Please define at least one of" \
			"    Eigen_ROOT (build/install directory), or" \
			"    Eigen_SRC  (source directory)" \
			"Exiting..." >&2
	fi
}

set_kokkidio_root () {
	local buildtype=$1

	Kokkidio_ROOT="$(set_root KOKKIDIO_INST "$sd/_install")"
	Kokkidio_ROOT+="/${backend}/$buildtype"
}

# check_install () {
# 	local subj=$1
# 	local testfile_tail="$2"
# 	local rootdir="${subj}_ROOT"
# 	echo "Checking $subj install..."
# 	for subdir in lib lib64; do
# 		local testfile="${!rootdir}/$subdir/$testfile_tail"
# 		echo "Checking for $subj cmake config file \"$testfile\"..."
# 		if [ -f "$testfile" ]; then
# 			echo "Found. ${rootdir} dir: ${!rootdir}"
# 			return
# 		else
# 			echo "Not found."
# 		fi
# 	done

# 	echo "Could not find $subj cmake config file! Did you install after building?"
# 	exit
# }

check_kokkos_install () {
	echo "Checking Kokkos install..."
	local testfile_tail="cmake/Kokkos/KokkosConfig.cmake"
	echo "Checking for Kokkos cmake config"
	for subdir in lib lib64; do
		local testfile="$Kokkos_ROOT/$subdir/$testfile_tail"
		echo -n "    \"$kk_testfile\": "
		if [ -f "$testfile" ]; then
			printf '%s\n%s\n' "Found." "Kokkos_ROOT dir: $Kokkos_ROOT"
			return
		else
			echo "Not found."
		fi
	done

	echo "Could not find Kokkos cmake config file! Did you install after building?"
	exit
}

check_eigen_install () {
	echo "Checking Eigen install..."
	local testfile="${Eigen_INST:-$Eigen_ROOT/build}/Eigen3Config.cmake"
	echo "Checking for Eigen cmake config"
	echo -n "    \"$testfile\": "
	if [ -f "$testfile" ]; then
		printf '%s\n%s\n' "Found." "Eigen_ROOT dir: $Eigen_ROOT"
	else
		echo "Not found." >&2
		exit
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

reset_cmake_flags () {
	# reset cmakeFlags between runs/backends etc.
	cmakeFlags=""
	cmakeFlags+=" -DCMAKE_BUILD_TYPE=$buildtype"
	# cmakeFlags+=" -DCMAKE_CXX_STANDARD=17"
	# Kokkos otherwise emits a warning
	cmakeFlags+=" -DCMAKE_CXX_EXTENSIONS=OFF"
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

