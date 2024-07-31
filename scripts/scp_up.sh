#!/bin/bash

keyfile="$HOME/.ssh/hlrn_ed25519"
host=beitritt@glogin-gpu.hpc.gwdg.de
targetDir=kokkidio

# Array of directories, files, and patterns to exclude
exclude=("_*" ".*" "media/" "timings/" "Doxyfile" "*LICENSE*" "*README*")

# script path
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

process_item() {
	local item="$1"
	for excl in "${exclude[@]}"; do
		if { [ -d "$item" ] && [[ "$item/" == $excl ]] ;} || [[ "$item" == $excl ]]; then
			return
		fi
	done

	echo "Current item: $item"
	scp -i "$keyfile" -r "$item" ${host}':~/'$targetDir
}

for item in * .*; do
	process_item "$item"
done
