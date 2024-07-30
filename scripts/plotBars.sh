#!/bin/bash

# script path
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cat | gnuplot -p "${sd}/plotBars.gp"
