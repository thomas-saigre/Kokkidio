#!/bin/bash

set -euv

# Followed gcc wiki on offloading, section "Building and Obtaining GCC" -> Linux distributions -> Debian/Ubuntu
# https://gcc.gnu.org/wiki/Offloading
# sudo apt install gcc-13 g++-13 gcc-13-offload-nvptx
# this also installs the distro package for nvptx-tools and libgomp-plugin-nvptx1
# you may also have to do the whole update-alternatives thing to get gcc-13 to be the default -
# or you specify the version when calling it.
# compile command:
# g++ omp_offloading.cpp -o omp_offloading.bin -fopenmp -foffload=nvptx-none -fcf-protection=none -foffload=-misa=sm_35

# about the error "ptxas fatal   : Value 'sm_30' is not defined for option 'gpu-name'":
# https://groups.google.com/g/linux.debian.bugs.dist/c/vbJUSEbhypE
# From there:
# "That said, with the upcoming GCC 13 you'll be able to build (!) GCC/nvptx
# with a '--with-arch=[...]' 'configure' option, see
# <https://gcc.gnu.org/gcc-13/changes.html#nvptx>. "

# So instead we build from source. Used references:
# https://gcc.gnu.org/wiki/Offloading
# https://gcc.gnu.org/install/specific.html#nvptx-x-none

isLocal=true
isLocal=false

if [[ $isLocal == true ]]; then
	INSTDIR=$HOME/pkg/omp-gcc
	WORKDIR=$INSTDIR/src
	nproc=32
else
	WORKDIR=/dev/shm/__`whoami`
	INSTDIR=/scratch-emmy/usr/`whoami`/omp-gcc
	nproc=40
	# I think using a somewhat modern GCC is optional, but could be helpful
	# cuda is requried
	module load cuda/12.1 gcc/9.3.0
fi

mkdir -p $WORKDIR && cd $WORKDIR

# Let's perform both downloads before potentially switching to a compute node
# nvptx-tools first, which we'll need before building gcc
git clone https://github.com/SourceryTools/nvptx-tools

# now let's get gcc. To make it reproducible, let's grab an actual release.
# downloading takes a minute
wget https://ftp.gwdg.de/pub/misc/gcc/releases/gcc-13.2.0/gcc-13.2.0.tar.xz

# We also need Newlib. That's why the error "C compiler cannot create executables"
# occurred before.
# this says so: https://gcc.gnu.org/wiki/Offloading, under Building and obtaining GCC
NEWLIBVER=newlib-4.4.0.20231231
# somehow, the HLRN fails to download this file.
# I recommend downloading it yourself and scp-ing it over to the HLRN.
# Alternatively, you could use git clone, but that's extremely slow.
wget ftp://sourceware.org/pub/newlib/$NEWLIBVER.tar.gz

# extracting GCC can take about 25min on scratch-emmy,
# but is very fast in a ramdisk such as /dev/shm (same at home)
tar -xf gcc-13.2.0.tar.xz
tar -xzf $NEWLIBVER.tar.gz

cd $WORKDIR/gcc-13.2.0
# there are some missing dependencies
./contrib/download_prerequisites

# let Newlib be built together with gcc (very important!)
ln -s $WORKDIR/$NEWLIBVER/newlib .

# now we do the compilations.
cd $WORKDIR/nvptx-tools
# this takes a little bit
mkdir build && cd build
../configure --prefix=$INSTDIR/nvptx-tools
# but the actual compilation is fast
make -j$nproc && make install

# we need to create two builds, one for the host, one for the device.
# first, the GPU build
cd $WORKDIR/gcc-13.2.0
mkdir build-nv && cd build-nv

../configure --prefix=$INSTDIR/gcc-13.2.0 \
	--with-build-time-tools=$INSTDIR/nvptx-tools/nvptx-none/bin \
	--target=nvptx-none --with-arch=sm_70 \
	--enable-as-accelerator-for=x86_64-pc-linux-gnu

# then, the host build
cd $WORKDIR/gcc-13.2.0
mkdir build-gcc && cd build-gcc

../configure --prefix=$INSTDIR/gcc-13.2.0 --enable-offload-targets=nvptx-none --disable-multilib

cd ../build-nv
make -j$nproc && make install

cd ../build-gcc
make -j$nproc && make install

for bin in ar as ld nm ranlib; do
	ln -s \
		$INSTDIR/nvptx-tools/bin/nvptx-none-$bin \
		$INSTDIR/gcc-13.2.0/libexec/gcc/x86_64-pc-linux-gnu/13.2.0/accel/nvptx-none/$bin
done

echo "Now you can call $INSTDIR/gcc-13.2.0/bin/gcc"
$INSTDIR/gcc-13.2.0/bin/gcc --version

echo "You may delete the contents of $WORKDIR now with"
echo "rm -rf $WORKDIR"

if [[ isLocal != true ]]; then
	rm -rf $WORKDIR
fi
