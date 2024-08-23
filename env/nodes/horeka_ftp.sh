
backend_default=hip
PKGDIR="$HOME/pkg"
source "${env_dir}/pkgdir_shared.sh"

# AMD Instinct MI100
Kokkos_ARCH=Kokkos_ARCH_AMD_GFX908

# recommendations from 
# https://www.nhr.kit.edu/userdocs/ftp/amd-rocm/
module load toolkit/rocm 
# module load compiler/gnu
module load devel/cmake

ROCM_DIR="/opt/rocm-6.1.0"
cmakeFlags_add="-DCMAKE_PREFIX_PATH=$ROCM_DIR"

# export CXX=`command -v hipcc`
export PATH=$(prepend_path "$ROCM_DIR/lib/llvm/bin" "${PATH:-}")

# echo "which clang: " `which clang`
export CXX=`command -v clang++`
export CC=`command -v clang`
