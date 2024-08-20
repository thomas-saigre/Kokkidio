add_executable( norm "" )

target_sources( norm PRIVATE
	main.cpp
	norm_cpu.cpp
	norm_unif_cpu.cpp
)

set_is_cpu( 
	main.cpp
	norm_cpu.cpp
	norm_unif_cpu.cpp
)

if (KOKKIDIO_USE_CUDA OR KOKKIDIO_USE_HIP)
	target_sources( norm PRIVATE
		norm_cuda_hip.cpp
	)
endif()

if(NOT KOKKIDIO_CPU_ONLY)
	target_sources( norm PRIVATE
		norm_unif_gpu.cpp
	)
endif()

kokkidio_configure_target(norm)
set_warnings(norm)
