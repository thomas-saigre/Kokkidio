#ifndef KOKKIDIO_UNIFY_CUDA_HIP_HPP
#define KOKKIDIO_UNIFY_CUDA_HIP_HPP

#ifndef KOKKIDIO_UNIFYBACKENDS_PUBLIC_HEADER
#error "Do not include this file directly! Include unifyBackends.hpp instead!"
#endif

/* including Eigen/Dense (via typeAliases.hpp) works,
 * if we do so before including the hip runtime. */
#include "Kokkidio/typeAliases.hpp"

#if defined(KOKKIDIO_USE_CUDA)
#include <cuda_runtime.h>
#elif defined(KOKKIDIO_USE_HIP)
#include <hip/nvidia_detail/nvidia_hip_runtime.h> // defines __HIPCC__ it seems.
#include <hip/nvidia_detail/nvidia_hip_runtime_api.h>
#endif

/* including Eigen/Dense (via typeAliases.hpp) after the hip runtime
 * means that both __NVCC__ and __HIPCC__ are #defined,
 * which triggers an #error in Eigen's Macros.h:511:
 * "NVCC as the target platform for HIPCC is currently not supported." */
// #include "Kokkidio/typeAliases.hpp"

// #include <utility>

/* Macro for error handling (wraps every call to CUDA function) */
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

/* "ch" is CudaHip */
#if defined(KOKKIDIO_USE_CUDA)
// #define ch cuda // could have required users to write ch##Error_t etc.
#define chError_t cudaError_t
#define chSuccess cudaSuccess
#define chGetErrorString cudaGetErrorString
#define chMalloc cudaMalloc
#define chFree cudaFree
#define chMemcpy cudaMemcpy
#define chMemcpyKind cudaMemcpyKind
#define chMemcpyHostToDevice cudaMemcpyHostToDevice
#define chMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define chGetLastError cudaGetLastError
#define chDeviceSynchronize cudaDeviceSynchronize
#define chLaunchKernel(FUNC_NAME, DIM_GRID, DIM_BLOCK, SHARED_BYTES, STREAM, ...) \
FUNC_NAME<<<DIM_GRID, DIM_BLOCK, SHARED_BYTES, STREAM>>>(__VA_ARGS__)
#elif defined(KOKKIDIO_USE_HIP)
// #define ch hip
#define chError_t hipError_t
#define chSuccess hipSuccess
#define chGetErrorString hipGetErrorString
#define chMalloc hipMalloc
#define chFree hipFree
#define chMemcpy hipMemcpy
#define chMemcpyKind hipMemcpyKind
#define chMemcpyHostToDevice hipMemcpyHostToDevice
#define chMemcpyDeviceToHost hipMemcpyDeviceToHost
#define chGetLastError hipGetLastError
#define chDeviceSynchronize hipDeviceSynchronize
#define chLaunchKernel hipLaunchKernelGGL
#endif


namespace Kokkidio
{

/* Handle errors encountered when invoking CUDA/HIP functions */
static void HandleError(chError_t err, const char *file, int line){
	if (err != chSuccess){
		fprintf(stderr, "%s in %s at line %d\n", chGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

inline void gpuFree(void* ptr){
	HANDLE_ERROR(chFree(ptr));
}

template<typename Scalar>
void gpuAlloc( Scalar*& dev_data, Index nElements ){
	HANDLE_ERROR(chMalloc((void**) &dev_data, sizeof(Scalar) * nElements ));
}

template<typename T>
void gpuAlloc( typename T::Scalar*& dev_data, const Eigen::DenseBase<T>& hostData ){
	gpuAlloc( dev_data, hostData.derived().size() );
	// HANDLE_ERROR(chMalloc((void**) &dev_data, sizeof(Scalar) * hostData.derived().size() ));
}

template<typename T>
void gpuMemcpyHostToDevice(
	typename T::Scalar*& dev_data,
	const Eigen::DenseBase<T>& hostData
){
	using Scalar = typename T::Scalar;
	HANDLE_ERROR(chMemcpy(
		dev_data,
		hostData.derived().data(),
		hostData.derived().size() * sizeof(Scalar),
		chMemcpyHostToDevice
	));
}

template<typename T>
void gpuMemcpyDeviceToHost(
	typename T::Scalar* const & dev_data,
	Eigen::DenseBase<T>& hostData
){
	using Scalar = typename T::Scalar;
	HANDLE_ERROR(chMemcpy(
		hostData.derived().data(),
		dev_data,
		hostData.derived().size() * sizeof(Scalar),
		chMemcpyDeviceToHost
	));
}

template<typename T>
void gpuAllocAndCopy( typename T::Scalar*& dev_data, const Eigen::DenseBase<T>& hostData ){
	gpuAlloc(dev_data, hostData);
	gpuMemcpyHostToDevice(dev_data, hostData);
}

} // namespace Kokkidio

#endif