#!/bin/bash

set -euv

# INSTDIR=$HOME/pkg/kokkos
# WORKDIR=$INSTDIR/src

# mkdir -p $WORKDIR && cd $WORKDIR

tag=4.3.00
archivename=$tag.tar.gz

wget https://github.com/kokkos/kokkos/archive/refs/tags/$archivename

tar -xf $archivename

rm $archivename
