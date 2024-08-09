cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(Kokkidio_examples)

find_package(Kokkidio 1.0 REQUIRED)
message(STATUS "Kokkidio_DIR: ${Kokkidio_DIR}")

message(STATUS "Using CXX compiler: ${CMAKE_CXX_COMPILER}")

macro(set_warnings TARGET_NAME)
	target_compile_options( ${TARGET_NAME} PRIVATE
		-Wall -Wextra -Wno-deprecated-declarations
	)
endmacro()

macro(configure TARGET_NAME)
	set_warnings(${TARGET_NAME})
	kokkidio_configure_target(${TARGET_NAME})
endmacro()

include_directories(
	"${CMAKE_SOURCE_DIR}/../../lib"
)

add_executable(viewmap ViewMap.cpp)

configure(viewmap)

add_executable(dualviewmap DualViewMap.cpp)

configure(dualviewmap)

add_executable(dot dot.cpp)
set_is_cpu(dot.cpp)
configure(dot)

add_executable(axpy axpy.cpp)
configure(axpy)


