download () (
	download_impl () {
		local url="$1"
		local srcvar="$2"
		local srcdir="$3"
		if [ -n "$srcdir" ]; then
			git clone "$url" "${srcdir}"
		else
			echo "${srcvar} not set! Please set it in your node file, e.g."
			echo "${nodefile}"
		fi
	}
	if [[ $buildKokkos == true ]]; then
		download_impl https://github.com/kokkos/kokkos.git Kokkos_SRC "${Kokkos_SRC+x}"
	fi
	if [[ $buildEigen == true ]]; then
		download_impl https://gitlab.com/libeigen/eigen.git Eigen_SRC "${Eigen_SRC+x}"
	fi
)