add_executable( dotProduct "" )

target_sources( dotProduct PRIVATE
	main.cpp
	dotProduct_cpu.cpp
	dotProduct_cpu_omp.cpp
	# dotProduct_cpu_omp_rng.cpp
	dotProduct_unif_cpu.cpp
	dotProduct_unif_gpu.cpp
)

set_is_cpu( 
	main.cpp
	dotProduct_cpu.cpp
	dotProduct_cpu_omp.cpp
	# dotProduct_cpu_omp_rng.cpp
	dotProduct_unif_cpu.cpp
)

if (KOKKIDIO_USE_CUDA OR KOKKIDIO_USE_HIP)
	target_sources( dotProduct PRIVATE
		# dotProduct_gpu_cuda_hip.cpp
		dotProduct_cuda_hip.cpp
	)
endif()

if (KOKKIDIO_USE_OMPT)
	target_sources( dotProduct PRIVATE 
		dotProduct_gpu_ompt.cpp
	)
endif()

conf(dotProduct)

