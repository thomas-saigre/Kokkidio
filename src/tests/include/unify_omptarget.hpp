#ifndef KOKKIDIO_UNIFY_OMPTARGET_HPP
#define KOKKIDIO_UNIFY_OMPTARGET_HPP

#ifndef KOKKIDIO_UNIFYBACKENDS_PUBLIC_HEADER
#error "Do not include this file directly! Include unifyBackends.hpp instead!"
#endif

/* including Eigen/Dense (via typeAliases.hpp) works,
 * if we do so before including the hip runtime. */
#include "Kokkidio/typeAliases.hpp"

#include <omp.h>
#include <cassert>

namespace Kokkidio
{

template<typename T>
void gpuAlloc(
	typename T::Scalar*& dev_data,
	const Eigen::DenseBase<T>& hostData
){
	using Scalar = typename T::Scalar;
	dev_data = static_cast<Scalar*>( omp_target_alloc(
		hostData.derived().size() * sizeof(Scalar),
		omp_get_default_device()
	) );
	assert(dev_data);
}

template<typename T>
void gpuMemcpyHostToDevice(
	typename T::Scalar*& dev_data,
	const Eigen::DenseBase<T>& hostData
){
	using Scalar = typename T::Scalar;
	omp_target_memcpy(
		dev_data,
		hostData.derived().data(),
		hostData.derived().size() * sizeof(Scalar), 0, 0,
		omp_get_default_device(),
		omp_get_initial_device()
	);
}

template<typename T>
void gpuMemcpyDeviceToHost(
	typename T::Scalar* const & dev_data,
	Eigen::DenseBase<T>& hostData
){
	using Scalar = typename T::Scalar;
	omp_target_memcpy(
		hostData.derived().data(),
		dev_data,
		hostData.derived().size() * sizeof(Scalar), 0, 0,
		omp_get_initial_device(),
		omp_get_default_device()
	);
}

template<typename T>
void gpuAllocAndCopy( typename T::Scalar*& dev_data, const Eigen::DenseBase<T>& hostData ){
	gpuAlloc(dev_data, hostData);
	gpuMemcpyHostToDevice(dev_data, hostData);
}

template<typename T>
void gpuFree( T*& dev_data ){
	omp_target_free( dev_data, omp_get_default_device() );
}

} // namespace Kokkidio

#endif