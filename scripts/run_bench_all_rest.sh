#!/bin/bash

set -eu

# script path
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

isQuickTest=${1:-false}

runscr="${sd}/run_bench.sh"

run_all () {
	local example=$1
	for target in cpu gpu; do
		if [[ "$target" == "gpu" ]] || [[ $isQuickTest == true ]]; then
			np=(4)
		else
			np=(1)
		fi
		for p in "${np[@]}"; do
			for scalar in float double; do
				# for b in cuda hip ompt sycl; do
				for b in cuda ompt sycl; do
					"$runscr" $b $scalar $example $target $p $isQuickTest
				done
			done
		done
	done
}

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# run_all dotProduct
run_all friction
