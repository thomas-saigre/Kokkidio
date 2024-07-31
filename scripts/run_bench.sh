#!/bin/bash

set -eu

# script path
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


print_help () {
echo -n "Valid options are:
  -h, --help            : Show options.
  -b, --backend         : Set backend. Valid options are
                          \"cuda\", \"hip\", \"sycl\", and \"ompt\".
  -n, --nCores          : Set number of cores (integer).
  -s, --scalar,         : Set floating point type. Valid options are
      --real              \"float\", and \"double\".
  -t, --target          : Set target. Valid options are
                          \"cpu\", and \"gpu\".
  -x, --example         : Which example to run. Valid options are
                          \"axpy\", \"dotProduct\", and \"friction\".
  -q, --quick           : Run quick test only.
"
}


backend=cuda
cpusPerTask=4
scalar=float
target=cpu
example=axpy
isQuickTest=false


! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
	echo "Enhanced getopt not available. Exiting..."
	exit
fi
VALID_ARGS=$(getopt \
-o b:n:s:t:x:hq \
--long help,backend:,nCores:,scalar:,real:,target:,example:,quick \
-- "$@")
if [[ $? -ne 0 ]]; then
	echo "getopt failed to parse arguments. Exiting..."
	exit 1;
fi

eval set -- "$VALID_ARGS"

while [ : ]; do
	case "$1" in
		-h | --help)
			printf '%s %s \n' \
				"Run test examples in a parameterised way, and write the runtimes to file."
			print_help
			exit
			;;
		-x | --example)
			if ! [[ "$2" =~ axpy|dotProduct|friction ]]; then
				echo "Unknown example: \"$2\". Exiting..."
				print_help
				exit
			fi
			example="$2"
			shift 2
			;;
		-b | --backend)
			if ! [[ "$2" =~ cuda|hip|sycl|ompt|cpu_gcc ]]; then
				echo "Unknown backend: \"$2\". Exiting..."
				print_help
				exit
			fi
			backend="$2"
			shift 2
			;;
		-s | -f | --scalar | --real)
			if ! [[ "$2" =~ float|double ]]; then
				echo "Unknown floating point type: \"$2\". Exiting..."
				print_help
				exit
			fi
			scalar="$2"
			shift 2
			;;
		-t | --target)
			if ! [[ "$2" =~ cpu|gpu ]]; then
				echo "Unknown target: \"$2\". Exiting..."
				print_help
				exit
			fi
			target="$2"
			shift 2
			;;
		-n | --nCores)
			if [[ $2 == ?(-)+([0-9]) ]]; then
				cpusPerTask=$2
			else
				echo "Option -n/--nCores must be an integer. Exiting..."
				print_help
				exit
			fi
			shift 2
			;;
		-q | --quick)
			isQuickTest=true
			shift 1
			;;
		--) shift;
			break
			;;
	esac
done

echo "Example:   $example"
echo "Backend:   $backend"
echo "Scalar:    $scalar"
echo "Target:    $target"
echo "nCores:    ${cpusPerTask}"
echo "quickTest: $isQuickTest"


if [[ "$target" == "gpu" ]]; then
	iter=10000
elif [[ "$target" == "cpu" ]]; then
	iter=100
else
	echo "Invalid target! Use \"gpu\" or \"cpu\"! Exiting..."
	exit
fi


export OMP_NUM_THREADS=$cpusPerTask
# Kokkos likes this better:
export OMP_PROC_BIND=spread
export OMP_PLACES=threads


rows=4
if [[ "$example" != "dotProduct" ]]; then
	rows=(1)
fi

cols=(
	10
	20
	50
	100
	200
	500
	1000
	2000
	5000
	10000
	20000
	50000
	100000
	200000
	500000
	1000000
	2000000
	5000000
	10000000
	20000000
	50000000
	100000000
)

if [ $isQuickTest = true ]; then
	cols=(10000 100000)
fi


bin="${sd}/../_build/tests/$backend/Release/${scalar}/${example}/${example}"

if [[ ! -f "$bin" ]]; then
	echo "Binary \"$bin\" does not exist! Exiting..."
	exit 1
fi


sep="	"

outdir="${sd}/../timings/`uname -n`/`date --iso-8601`/${scalar}"
outfile="$outdir/${example}_${backend}_${scalar}"
if [[ "$target" == "gpu" ]]; then
	outfile+="_gpu"
else
	outfile+="_${cpusPerTask}p"
fi
# outfile="$outdir/${example}_${backend}_${scalar}_${cpusPerTask}p"

echo "output directory: ${outdir}"

mkdir -p ${outdir}

if [ $isQuickTest = true ]
then
	outfile+="_quick_test"
fi
logfile="$outfile.log"
outfile+=".csv"
echo "Outfile: $outfile"
		
optkeys=()

if [[ "$example" == "axpy" ]]; then
	if [[ "$target" != "cpu" ]]; then
		if [[ "$backend" =~ cuda|hip ]]; then
			optkeys+=(
				"GPU--native-cstyle"
			)
		fi
		optkeys+=(
			"GPU--unified-cstyle"
			"GPU--unified-kokkos"
			"GPU--unified-kokkidio"
		)
	fi
	if [[ "$target" != "gpu" ]]; then
		optkeys+=(
			"CPU--unified-warmup"
			"CPU--unified-cstyle"
			"CPU--unified-kokkos"
			"CPU--unified-kokkidio"
			"CPU--native-cstyle_parallel"
			"CPU--native-cstyle_parallel"
			"CPU--native-cstyle_sequential"
			"CPU--native-eigen_sequential"
			"CPU--native-eigen_parallel"
		)
	fi
fi

if [[ "$example" == "dotProduct" ]]; then
	if [[ "$target" != "cpu" ]]; then
		if [[ "$backend" =~ cuda|hip ]]; then
			optkeys+=(
				"GPU--native-cstyle_blockbuf"
			)
		fi
		optkeys+=(
			"GPU--unified-cstyle_stackbuf"
			"GPU--unified-cstyle_nobuf"
			"GPU--unified-eigen_colwise"
			"GPU--unified-eigen_ranged"
			# "GPU--unified-eigen_ranged_for_each"
			# "GPU--unified-eigen_colwise_merged"
			# "GPU--unified-eigen_anged_for_each_merged"
		)
	fi
	if [[ "$target" != "gpu" ]]; then
		optkeys+=(
			"CPU--unified-cstyle_stackbuf"
			"CPU--unified-cstyle_nobuf"
			"CPU--unified-eigen_colwise"
			"CPU--unified-eigen_ranged"
			# "CPU--unified-eigen_ranged_for_each"
			# "CPU--unified-eigen_colwise_merged"
			# "CPU--unified-eigen_anged_for_each_merged"
			"CPU--native-seq_cstyle"
			# "CPU--native-seq_manual"
			"CPU--native-seq_eigen_colwise"
			"CPU--native-seq_eigen_arrProd"
			"CPU--native-par_cstyle"
			# "CPU--native-par_manual"
			"CPU--native-par_eigen_colwise"
			"CPU--native-par_eigen_arrProd"
		)
	fi
else
	rows=(1)
fi

if [[ "$example" == "friction" ]]; then
	if [[ "$target" != "cpu" ]]; then
		if [[ "$backend" =~ cuda|hip ]]; then
			optkeys+=(
				"GPU--native-cstyle "
				# "GPU--native-colwise_fullbuf"
			)
		fi
		optkeys+=(
			"GPU--unified-cstyle"
			"GPU--unified-eigen_colwise_fullbuf"
			"GPU--unified-eigen_colwise_stackbuf"
			"GPU--unified-eigen_ranged_fullbuf"
			"GPU--unified-eigen_ranged_chunkbuf "
			# "GPU--unified-context_ranged"
		)
	fi
	if [[ "$target" != "gpu" ]]; then
		optkeys+=(
			"CPU--unified-cstyle"
			"CPU--unified-eigen_colwise_fullbuf"
			"CPU--unified-eigen_colwise_stackbuf"
			"CPU--unified-eigen_ranged_fullbuf"
			"CPU--unified-eigen_ranged_chunkbuf"
			# "CPU--unified-context_ranged"
			"CPU--native-cstyle"
			# "CPU--native-eigen_colwise_fullbuf"
		)
	fi
fi

writeCol () {
	local outfile=$1
	local entry=$2
	sed -i '$s/$/'"$sep$entry"'/' "$outfile"
}

writeColTitles () {
	local outfile="$1"
	local colTitles="$2"
	local i
	for key in "${optkeys[@]}"
	do
		key="${key//[, ]/-}"
		colTitles+="${sep}${key}"
	done
	echo "`basename $1`"$'\n'"$colTitles" > $outfile
}

grepTiming () {
	local key=$2
	local lines=`echo "$1" | grep -A 1 "Run: $key"`
	# echo "lines=\"$lines\""
	if [ "$lines" ]; then
		local t=`echo $lines | \
			grep "Computation time" | \
			sed 's/^.*time: \(.*\) seconds\./\1/'`
	else
		local t="#NV"
	fi
	# t=$(echo "$1" | \
	# 	grep -A 1 "Run: $key" | \
	# 	grep "Computation time" | \
	# 	sed 's/^.*time: \(.*\) seconds\./\1/')
	echo $t
}

writeColTitles "$outfile" "iter${sep}cols"

for col in "${cols[@]}"
do
	echo "$iter$sep$col" >> "$outfile"
	echo "cols: ${col}, iterations: ${iter}"
	if [[ "$example" == "dotProduct" ]]; then
		output="$("$bin" -s $rows $col -r $iter -t $target)"
	else
		output="$("$bin" -s $col -r $iter -t $target)"
	fi
	echo "$output" >> "$logfile"

	i=0
	for run in "${optkeys[@]}"; do
		t_run=$(grepTiming "$output" "${optkeys[$i]}")
		echo "	-> time (${optkeys[$i]}): $t_run"
		writeCol "$outfile" $t_run
		((++i))
	done
done

echo "Done."
