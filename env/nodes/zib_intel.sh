
module load intel
backend_default=sycl

PKGDIR="$HOME/pkg"
source "${env_dir}/pkgdir_shared.sh"

Kokkos_ARCH=Kokkos_ARCH_INTEL_PVC

# export CXX=`command -v clang++`
export CC=`command -v icx`
export CXX=`command -v icpx`
