
module load intel
backend_default=sycl

PKGDIR="$HOME/pkg"
source "${env_dir}/pkgdir_shared.sh"

Kokkos_ARCH=Kokkos_ARCH_INTEL_PVC

# cmakeFlags+=" -DCMAKE_CXX_COMPILER=$(command -v clang++)"
cmakeFlags+=" -DCMAKE_CXX_COMPILER=$(command -v icpx)"
