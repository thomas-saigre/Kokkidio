
print_help () {
echo -n "Synopsis:
  build.sh [options] [<subject>]

  <subject>             : Specifies what to build. Can be
                          \"kokkidio\" (default),
                          \"examples\",
                          \"tests\", or
                          \"kokkos\".
  kokkidio              : Build Kokkidio. Requires Eigen and Kokkos to be 
                          available via CMake's find_package.
  examples              : Build Kokkidio example executables. Requires Kokkidio
                          to be built first.
  tests                 : Build the Kokkidio tests.
  kokkos                : Convenience option to let this script build Kokkos.
                          Requires setting the environment variable 
                          Kokkos_SRC
                          to the directory containing the Kokkos source files.
                          You may use env/env_tpl.sh for this.
  -h, --help            : Show options.
  -b, --backend         : Set backend. Valid options are
                          \"cuda\", \"hip\", \"sycl\", \"ompt\", and \"all\".
                          Backend-specific settings (compiler, arch, etc.)
                          must be provided in the respective file:
                          env/env_<backend>.sh
                          Applicable to all subjects.
  -c, --compile-only,   : Don't execute after compilation. Useful when
      --no-run            compilation and execution occur on separate systems.
                          Only affects tests.
  -n, --no-compilation, : Configuration only, no compilation. Excludes option
      --no-compile        --install.
                          Applicable to Kokkos and Kokkidio tests.
  -i, --install         : Install after configuration, so that it can be found 
                          by CMake projects, using find_package.
                          Applicable to Kokkos and Kokkidio.
  -p, --prefix          : Install prefix for Kokkidio. Implies --install.
                          Default is <source directory>/install.
                          Kokkos install prefix must be set via environment var
                          Kokkos_INST instead.
  -s, --scalar,         : Set floating point type. Valid options are
      --real              \"float\", \"double\", and \"all\".
                          Only affects tests.
  -t, --build-type      : Set build type. Valid options are
                          \"Debug\", \"Release\", and \"all\".
                          Applicable to all subjects.

The default file for specifying machine-specific variables is:
    <kokkidio>/env/nodes/${node_name}.sh
but this may be overriden in 
    <kokkidio>/env/node_patterns.sh
When these variables are correctly configured, the following command builds all components:
    ./build.sh -ic kokkos kokkidio examples tests
"
}

noBuild="false"
noRun="false"
backend="default"
whichScalar="all"
whichBuildtype="all"
install_opt="false"
install_prefix=""

check_backend () {
	if ! [[ "$1" =~ all|cuda|hip|sycl|omp|cpu_gcc ]]; then
		echo "Unknown backend: \"$1\". Exiting..."
		print_help
		exit
	fi
}

! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
	echo "Enhanced getopt not available. Exiting..."
	exit
fi
VALID_ARGS=$(getopt \
-o b:s:t:p:hinc \
--long help,backend:,scalar:,real:,prefix:,build-type:,install,no-compilation,no-compile,compile-only,no-run \
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
				"Build script for Kokkidio and its tests." \
				"Intended to allow multiple backends on the same machine."
			print_help
			exit
			;;
		-b | --backend)
			check_backend "$2"
			backend="$2"
			echo "Backend: \"$2\"."
			shift 2
			;;
		-p | --prefix)
			if [ "$2" = /* ]; then
				install_prefix="$(pwd)/$2"
			else
				install_prefix="$2"
			fi
			echo "Install prefix: \"$install_prefix\"."
			install_opt=true
			shift 2
			;;
		-s | -f | --scalar | --real)
			if ! [[ "$2" =~ all|float|double ]]; then
				echo "Unknown floating point type: \"$2\". Exiting..."
				print_help
				exit
			fi
			whichScalar="$2"
			echo "Floating point type: \"$2\"."
			shift 2
			;;
		-t | --build-type)
			if ! [[ "$2" =~ all|Release|Debug ]]; then
				echo "Unknown build type: \"$2\". Exiting..."
				print_help
				exit
			fi
			whichBuildtype="$2"
			echo "Build type: \"$2\"."
			shift 2
			;;
		-i | --install)
			install_opt=true
			shift 1
			;;
		-n | --no-compilation | --no-compile)
			noBuild=true
			shift 1
			;;
		-c | --compile-only | --no-run)
			noRun=true
			shift 1
			;;
		--) shift;
			break
			;;
	esac
done

subjects=()

if [ $# -eq 0 ]; then
	subjects+=("kokkidio")
else
	while [ $# -gt 0 ]; do
		subjects+=(${1,,})
		shift
	done
fi

buildKokkidio=false
buildTests=false
buildExamples=false
buildKokkos=false

for subj in "${subjects[@]}"; do
	case "$subj" in
		"kokkidio")
			buildKokkidio=true
			;;
		"examples")
			buildExamples=true
			;;
		"tests")
			buildTests=true
			;;
		"kokkos")
			buildKokkos=true
			;;
		*)
			echo "Unknown subject: \"${subj}\""
			print_help
			exit
			;;
	esac
done

if [[ $install_opt == true ]] && [[ $buildKokkidio != true ]] && [[ $buildKokkos != true ]]; then
	echo "Tests cannot be installed. Ignoring options --install and --prefix."
fi

printf 'Building: '
subj_isFirst=true
for subj in "${subjects[@]}"; do
	if [[ $subj_isFirst != true ]]; then
		printf ', '
	fi
	subj_isFirst=false
	printf '%s' $subj
done
printf '\n'

if [[ $whichBuildtype == "all" ]]; then
	buildtypes=("Debug" "Release")
else
	buildtypes=("$whichBuildtype")
fi
