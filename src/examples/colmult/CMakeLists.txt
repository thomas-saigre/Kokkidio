
add_executable(colmult "")
target_sources(colmult PRIVATE
	colmult_main.cpp
	colmult_host.cpp
	colmult_device.cpp
)

set_is_cpu(
	colmult_host.cpp
)

configure(colmult)

add_executable(colmult_eigen colmult_eigen.cpp)

set_is_cpu(
	colmult_eigen.cpp
)

configure(colmult_eigen)
