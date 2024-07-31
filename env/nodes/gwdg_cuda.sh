
module load cmake gcc cuda
backend_default=cuda

PKGDIR="/scratch-emmy/usr/`whoami`"
source "${env_dir}/pkgdir_shared.sh"

Kokkos_ARCH=Kokkos_ARCH_AMPERE80
# Kokkos_ARCH=Kokkos_ARCH_HOPPER90
