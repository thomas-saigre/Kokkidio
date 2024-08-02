#ifndef KOKKIDIO_MapView_HPP
#define KOKKIDIO_MapView_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Core.hpp instead."
#endif

#include "Kokkidio/TargetSpaces.hpp"
#include "Kokkidio/typeAliases.hpp"
#include "Kokkidio/util.hpp"
#include "Kokkidio/EigenTypeHelpers.hpp"
#include "Kokkidio/memory.hpp"
#include "Kokkidio/syclify_macros.hpp"

#include <Kokkos_Core.hpp>

namespace Kokkidio
{

template<typename _PlainObjectType, Target targetArg = DefaultTarget>
class MapView {
public:
	static constexpr Target target { ExecutionTarget<targetArg> };
	using PlainObjectType_host = _PlainObjectType;
	/* To make the MapView work, the device view must be non-const in most 
	 * cases, to not end up with inaccessible device memory.
	 * However, if the target is the host, then a cast to non-const
	 * could allow write access to a const object. 
	 * Instead, any allocations and copies are simply skipped in that case.
	 * */
	using PlainObjectType_target = std::conditional_t<target == Target::host,
		PlainObjectType_host,
		std::remove_const_t<PlainObjectType_host>
	>;

	using ThisType = MapView<PlainObjectType_target, target>;
	using MemorySpace    = Kokkidio::MemorySpace   <target>;
	using ExecutionSpace = Kokkidio::ExecutionSpace<target>;
private:
	using ViewTypeStruct = Kokkidio::detail::ViewType<PlainObjectType_target, MemorySpace>;
public:
	using ViewType   = typename ViewTypeStruct::Type;
	using Scalar     = typename ViewTypeStruct::Scalar;
	using HostMirror = typename ViewType::HostMirror;
	using MapType    = Eigen::Map<PlainObjectType_host>;

	static_assert( is_contiguous<PlainObjectType_target>() );

protected:
	ViewType m_view;
	observer_ptr<PlainObjectType_host> m_obj {nullptr};

public:

	/* For fixed size Eigen types,
	 * the default constructor allocates memory,
	 * while for dynamically sized Eigen objects, it does nothing,
	 * like in Eigen itself. */
	MapView(){
		/* MapView(Index, Index) overwrites rows/cols 
		 * if they're known at compile time,
		 * so we could pass any numbers. */
		using P = PlainObjectType_host;
		if constexpr ( P::SizeAtCompileTime != Eigen::Dynamic ){
			this->allocView(P::RowsAtCompileTime, P::ColsAtCompileTime);
		}
	}

	MapView(Index rows, Index cols){
		this->allocView(rows, cols);
	}

	/* For Eigen vector types,
	 * we allow a single size parameter, like in Eigen itself. */
	template<typename P = PlainObjectType_host,
		typename std::enable_if_t<P::IsVectorAtCompileTime, int> = 0>
	MapView(Index size) :
		/* MapView(Index, Index) overwrites rows/cols 
		 * if they're known at compile time,
		 * so we could pass any numbers. */
		MapView(size, size)
	{
		static_assert( std::is_same_v<P, PlainObjectType_host> );
	}

	MapView( PlainObjectType_host& hostObj ){
		this->wrapOrAlloc(hostObj);
	}

	/* cannot be called in device code */
	void resize(Index rows, Index cols){
		this->adjustDims(rows, cols);
		if ( rows == this->rows() && cols == this->cols() ){
			return;
		}
		if constexpr (
			!std::is_const_v<PlainObjectType_host> &&
			!is_eigen_map_v<std::remove_const_t<PlainObjectType_host>>
		){
			/* If the MapView was given a (non-const) object on construction,
			 * then MapView::resize should be the correct way to resize both,
			 * because the Eigen object and Kokkos::View don't know about each 
			 * other - i.e. there is no other non-manual way.
			 */
			if (m_obj){
				PlainObjectType_host& hostObj { *(this->m_obj) };
				/* Resize the host object */
				hostObj.resize(rows, cols);
				/* and if the target is the host, then the View is unmanaged,
				 * so we just wrap it around the host object */
				if constexpr ( target == Target::host ){
					this->wrapView(hostObj);
				/* On the device, the View is always managed, so we resize it */
				} else {
					this->resizeView(rows, cols);
				}
			/* Without a host object, it's also always a managed View */
			} else {
				this->resizeView(rows, cols);
			}
		} else {
			static_assert(dependent_false<PlainObjectType_host>::value && false,
				"Cannot resize a const or non-owning object!"
			);
		}
	}

	constexpr bool isManaged() const {
		/* The View is only unmanaged in one case:
		 * if a host object was provided during construction AND
		 * the target (~= ExecutionSpace) is also the host.
		 * A device View is always managed, 
		 * and so is a host View that doesn't wrap a host object. */
		return !(target == Target::host && m_obj);
	}

protected:
	void adjustRows(Index& rows) const {
		using T = PlainObjectType_target;
		if constexpr (T::RowsAtCompileTime != Eigen::Dynamic){
			rows = T::RowsAtCompileTime;
		}
	}

	void adjustCols(Index& cols) const {
		using T = PlainObjectType_target;
		if constexpr (T::ColsAtCompileTime != Eigen::Dynamic){
			cols = T::ColsAtCompileTime;
		}
	}

	void adjustDims(Index& rows, Index& cols) const {
		/* if the Eigen class has fixed size rows/columns,
		 * overwrite rows/cols */
		this->adjustRows(rows);
		this->adjustCols(cols);
	}

	void allocView(Index rows, Index cols){
		this->adjustDims(rows, cols);
		this->m_view = ViewType{
			Kokkos::view_alloc(MemorySpace{}, Kokkos::WithoutInitializing, ""),
			static_cast<std::size_t>(rows),
			static_cast<std::size_t>(cols)
		};
		printd( "(%p) Allocating View with address, size %i x %i.\n"
			, (void*) this->m_view.data()
			, static_cast<int>( this->rows() )
			, static_cast<int>( this->cols() )
		);
	}

	void resizeView(Index rows, Index cols){
		assert( this->isManaged() );
		// adjustDims(rows, cols);
		if ( rows == this->rows() && cols == this->cols() ){
			return;
		}
		Kokkos::resize(
			Kokkos::WithoutInitializing,
			this->m_view,
			static_cast<std::size_t>(rows),
			static_cast<std::size_t>(cols)
		);
		printd( "(%p) Setting view size to %i.\n", (void*) this, this->size() );
	}

	void wrapView(PlainObjectType_host& hostObj ){
		assert( !this->isManaged() );
		printd( "(%p) Creating View from data pointer, with size %i x %i.\n"
			, (void*) hostObj.data()
			, static_cast<int>( hostObj.rows() )
			, static_cast<int>( hostObj.cols() )
		);
		this->m_view = ViewType{ hostObj.data(),
			static_cast<std::size_t>( hostObj.rows() ),
			static_cast<std::size_t>( hostObj.cols() )
		};
		printd( "(%p) View now has size %i x %i.\n"
			, (void*) this->view().data()
			, static_cast<int>( this->rows() )
			, static_cast<int>( this->cols() )
		);
	}

	void wrapOrAlloc( PlainObjectType_host& hostObj ){
		if constexpr ( target == Target::host ){
			/* If the target is the host, then the memory is already accessible,
			 * Therefore, we don't need to allocate device memory, 
			 * and can simply wrap the host object's data in an unmanaged view.
			 * */
			this->m_obj = make_observer( &hostObj );
			this->wrapView(hostObj);
		} else {
			this->allocView( hostObj.rows(), hostObj.cols() );
		}
	}

public:

	KOKKOS_FUNCTION
	bool isAlloc() const {
		return this->m_view.is_allocated();
	}

	KOKKOS_FUNCTION
	Scalar* data() {
		assert( this->isAlloc() );
		return this->m_view.data();
	}

	KOKKOS_FUNCTION
	std::add_const_t<Scalar>* data() const {
		assert( this->isAlloc() );
		return this->m_view.data();
	}

	/**
	 * @brief Returns an Eigen::Map
	 * to memory on the MapView's \a target 
	 * (Target::host or Target::device).
	 * 
	 * This represents the core functionality of an MapView,
	 * because this Eigen::Map can be used in any Eigen operation.
	 * 
	 * If the MapView was initialised with an Eigen object \a obj and
	 * \a target == Target::host, then the Eigen::Map uses the same data 
	 * as that \a obj.
	 * Otherwise, it uses data of a managed Kokkos::View.
	 * 
	 * @return MapType 
	 */
	KOKKOS_FUNCTION
	auto map() const -> MapType {
		/* function must be marked as 'const',
		 * because the operator() of a lambda is 'const' by default,
		 * and thus a copy-capturing lambda will capture this class'
		 * 'this' pointer as const.
		 * */
		assert( isAlloc() );
		return { this->m_view.data(), this->rows(), this->cols() };
	}

	/**
	 * @brief Returns the stored Kokkos::View. 
	 * If the MapView was initialised with an Eigen object \a obj,
	 * this returns an unmanaged View with the same data pointer as \a obj.
	 * Otherwise, it returns a managed view, whose memory space is
	 * Kokkos::DefaultExecutionSpace, if \a target == Target::device, and
	 * Kokkos::DefaultHostExecutionSpace, if \a target == Target::host.
	 * 
	 * @return ViewType& 
	 */
	KOKKOS_FUNCTION
	auto view() const -> ViewType {
		return this->m_view;
	}

	KOKKOS_FUNCTION
	Index rows() const {
		return static_cast<Index>( this->m_view.extent(0) );
	}

	KOKKOS_FUNCTION
	Index cols() const {
		return static_cast<Index>( this->m_view.extent(1) );
	}

	KOKKOS_FUNCTION
	Index size() const {
		return static_cast<Index>( this->m_view.size() );
	}
};

// static_assert( std::is_trivially_copyable_v<MapView<ArrayXXs, Target::device>> );

template<typename T>
struct is_MapView : std::false_type {};

template<typename PlainObjectType, Target targetArg>
struct is_MapView<MapView<PlainObjectType, targetArg>> : std::true_type {};

template<typename T>
inline constexpr bool is_MapView_v = is_MapView<T>::value;

} // namespace Kokkidio

#endif
