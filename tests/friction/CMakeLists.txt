add_executable( friction "" )

target_sources( friction PRIVATE
	main.cpp
	friction_cpu.cpp
	friction_unif_cpu.cpp
	friction_context_cpu.cpp
)

set_is_cpu( 
	main.cpp
	friction_cpu.cpp
	friction_unif_cpu.cpp
	friction_context_cpu.cpp
)

if (KOKKIDIO_USE_CUDA OR KOKKIDIO_USE_HIP)
	target_sources( friction PRIVATE
		friction_gpu_cuda_hip.cpp
	)
endif()

if(NOT KOKKIDIO_CPU_ONLY)
	target_sources( friction PRIVATE
		friction_unif_gpu.cpp
		friction_context_gpu.cpp
	)
endif()

kokkidio_configure_target(friction)
set_warnings(friction)
