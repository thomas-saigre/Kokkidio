#!/bin/bash

set -eu

# script path
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

isQuickTest=${1:-false}
quickTestFlag=""
if [[ $isQuickTest == true ]]; then
	quickTestFlag="--quick"
fi

runscr="${sd}/run_bench.sh"

run_all () {
	local example=$1
	for target in gpu cpu; do
		if [[ "$target" == "gpu" ]] || [[ $isQuickTest == true ]]; then
			np=(4)
		else
			np=(1 2 4)
		fi
		for p in "${np[@]}"; do
			for scalar in float double; do
				# for b in cuda hip ompt sycl; do
				for b in cuda ompt sycl; do
					"$runscr" \
						--backend $b \
						--scalar $scalar \
						--example $example \
						--target $target \
						--nCores $p \
						$quickTestFlag
				done
			done
		done
	done
}

run_all axpy
# run_all dotProduct
# run_all friction
