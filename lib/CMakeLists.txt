
get_cmake_property(ALL_VARS VARIABLES)
message(STATUS "Active Kokkos variables:")
foreach(VAR ${ALL_VARS})
	if(VAR MATCHES "^Kokkos_ENABLE_(CUDA|HIP|SYCL|Kokkos_ENABLE_OPENMPTARGET)")
		if (VAR)
			message(STATUS "${VAR}=${${VAR}}")
		endif()
	endif()
endforeach()

add_subdirectory("kokkos-4.3.00" ${KOKKIDIO_KOKKOS_BUILDDIR})
