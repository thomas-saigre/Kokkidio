
find_package(Kokkos REQUIRED)

message(STATUS "Kokkos_DIR: ${Kokkos_DIR}")
message(STATUS "Kokkos_VERSION: ${Kokkos_VERSION}")
# # this is already printed by KokkosConfig.cmake:
# message(STATUS "Kokkos_DEVICES: ${Kokkos_DEVICES}")

macro(kokkidio_check_backend BACKEND_ARG KOKKIDIO_BACKEND_VAR)
	if ("${BACKEND_ARG}" IN_LIST Kokkos_DEVICES)
		set(${KOKKIDIO_BACKEND_VAR} ON)
		set(KOKKIDIO_BACKEND ${KOKKIDIO_BACKEND_VAR})
		message(STATUS "Backend: ${KOKKIDIO_BACKEND}")
	endif()
endmacro()

if (NOT DEFINED KOKKIDIO_BACKEND)
	kokkidio_check_backend(CUDA KOKKIDIO_USE_CUDA)
	kokkidio_check_backend(HIP KOKKIDIO_USE_HIP)
	kokkidio_check_backend(OPENMPTARGET KOKKIDIO_USE_OMPT)
	kokkidio_check_backend(SYCL KOKKIDIO_USE_SYCL)
endif()

if(KOKKIDIO_USE_CUDA)
	message(STATUS "Enabling CUDA as CMake language...")
	enable_language(CUDA)
	message(STATUS "Kokkos_CUDA_ARCHITECTURES: ${Kokkos_CUDA_ARCHITECTURES}")
endif()

