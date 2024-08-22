
module load cuda
# module load gcc/9.3.0
module load gcc # is v. 12.3 at time of writing
module load cmake/3.26.4

backend_default=cuda

PKGDIR="/scratch-emmy/usr/`whoami`"
source "${env_dir}/pkgdir_shared.sh"

Kokkos_ARCH=Kokkos_ARCH_AMPERE80
# Kokkos_ARCH=Kokkos_ARCH_HOPPER90

export CC=`command -v gcc`
export CXX=`command -v g++`
