

download () (
	download_impl () {
		local subj="$1"
		local url="$2"
		local srcvar="$3"
		local srcdir="$4"
		if [ -n "$srcdir" ]; then
			git clone "$url" "${srcdir}"
		else
			echo "${subj}_SRC not set! Please set it in your node file, e.g."
			echo "${nodefile}"
		fi
	}
	if [[ $buildKokkos == true ]]; then
		download_impl Kokkos https://github.com/kokkos/kokkos.git "${Kokkos_SRC+x}"
	fi
	if [[ $buildEigen == true ]]; then
		download_impl Eigen https://gitlab.com/libeigen/eigen.git "${Eigen_SRC+x}"
	fi
)


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