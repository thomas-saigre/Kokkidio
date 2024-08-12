
# set_if_defined(EIGEN_DIR EIGEN_ROOT)

# if(DEFINED EIGEN_DIR)
# 	find_package(Eigen3 3.4 REQUIRED HINTS ${EIGEN_DIR})
# else()
# 	find_package(Eigen3 3.4 REQUIRED NO_MODULE)
# 	message(STATUS "If you wish to supply a different version of Eigen, "
# 		"set the environment variable EIGEN_DIR to the path containing "
# 		"\"Eigen3Config.cmake\" or \"eigen3-config.cmake\", e.g.\n"
# 		"export EIGEN_DIR=$HOME/my_eigen/build"
# 	)
# endif()

set_if_defined(Eigen3_ROOT Eigen_ROOT)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
message(STATUS "Eigen library: ${Eigen3_DIR}")
message(STATUS "Eigen version: ${Eigen3_VERSION}")

function(set_is_cpu)
	if (KOKKIDIO_USE_CUDA)
		# When using CUDA, a Kokkos header can only be included in a CUDA file,
		# but Eigen generally disables vectorisation in CUDA files.
		# So by specifying which files are CPU only, 
		# and then tricking Eigen by #defining EIGEN_NO_CUDA in them,
		# we can include Kokkos headers AND have Eigen vectorise host code.
		set_source_files_properties( ${ARGV} PROPERTIES
			COMPILE_DEFINITIONS EIGEN_NO_CUDA
		)
	endif()
endfunction()
