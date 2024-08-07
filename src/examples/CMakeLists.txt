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

add_executable(mapview MapView.cpp)

configure(mapview)

add_executable(dualmapview DualMapView.cpp)

configure(dualmapview)
