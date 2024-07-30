#ifndef KOKKIDIO_OMPSEGMENT_HPP
#define KOKKIDIO_OMPSEGMENT_HPP

#ifndef KOKKIDIO_PUBLIC_HEADER
#error "Do not include this file directly. Include Kokkidio/Core.hpp instead."
#endif

#include "Kokkidio/ompSegment_base.hpp"
#include "Kokkidio/IndexRange.hpp"

namespace Kokkidio
{

template<typename Policy>
IndexRange<detail::IndexType<Policy>>
ompSegment( const Policy& pol ){
	return ompSegment( toIndexRange(pol) );
}


} // namespace Kokkidio


#endif