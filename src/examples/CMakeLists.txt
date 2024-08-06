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

include_directories(
	"${CMAKE_SOURCE_DIR}/../include"
	"${CMAKE_SOURCE_DIR}/../lib"
)

add_executable(mapview MapView.cpp)

set_warnings(mapview)

kokkidio_configure_target(mapview)
