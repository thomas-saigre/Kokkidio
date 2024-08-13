
# uncomment one of these
backend_default=cuda
# backend_default=hip
# backend_default=sycl
# backend_default=ompt

# see https://kokkos.org/kokkos-core-wiki/keywords.html#architectures
# for possible values
Kokkos_ARCH=Kokkos_ARCH_SOMEVALUE

# feel free to adjust this path to your liking
Kokkos_BASE="$HOME/pkg/kokkos"
# only Kokkos_SRC is required
Kokkos_SRC="$Kokkos_BASE/src/kokkos-dev"
# Kokkos_BUILD and Kokkos_INST are optional,
# but may be used to specify build/install directories for Kokkos.
Kokkos_BUILD="$Kokkos_BASE/build"
Kokkos_INST="$Kokkos_BASE/install"

Eigen_BASE="$HOME/pkg/eigen"
# only Eigen_SRC is required
Eigen_SRC="$Eigen_BASE"
# As a pure header library, Eigen doesn't need to be compiled.
# build and install directory can be the same, if you don't specifically wish 
# for the headers to be copied to standard directories.
Eigen_BUILD="$Eigen_BASE/build"
# Eigen_INST="$Eigen_BASE/install"
