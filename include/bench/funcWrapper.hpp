#ifndef KOKKIDIO_FUNC_WRAPPER_HPP
#define KOKKIDIO_FUNC_WRAPPER_HPP

#include <type_traits>
#include <array>
#include <functional>
#include "Kokkidio/TargetEnum.hpp"

namespace Kokkidio
{


template<template<Target, typename ImplEnum, ImplEnum, typename ...>class Functor>
struct make_func {
	template<Target target, typename ImplEnum, ImplEnum impl, typename ... Ts>
	decltype(auto) call(Ts&& ... args) const {
		static_assert( std::is_enum_v<ImplEnum> );
		return Functor<target, ImplEnum, impl, Ts...>()(
			std::forward<Ts>(args) ...
		);
	}

	template<Target target, typename ImplEnum, ImplEnum impl, typename ... Ts>
	decltype(auto) wrap(Ts&& ... args) const {
		static_assert( std::is_enum_v<ImplEnum> );
		using Ret = decltype( Functor<target, ImplEnum, impl, Ts...>()(
			std::forward<Ts>(args)...
		) );
		return std::function<Ret()>{
			[this, &args...]() {
				return this->template call<target, ImplEnum, impl>(
					std::forward<Ts>(args)...
				);
			}
		};
	}

	template<Target target, typename ImplEnum, ImplEnum ... impls, typename ... Ts>
	auto make_array(Ts&& ... args) const {
		static_assert( std::is_enum_v<ImplEnum> );
		// return std::array<std::function<Ret(Ts...)>, sizeof...(impls)>{
		// // return std::array<Ret(*)(), sizeof...(impls)>{
		// 	Functor<target, ImplEnum, impls, Ts...>() ...
		// };

		/* When using a parameter pack for both the enum
		 * and the function arguments,
		 * we need to wrap the function calls in a lambda.
		 * That way, the call can let the compiler deduce Ts 
		 * from the arguments to make_array only.
		 * */
		return std::array{ this->template wrap<target, ImplEnum, impls>(
			std::forward<Ts>(args)...
		) ... };
	}
};


#define KOKKIDIO_FUNC_WRAPPER(WRAPPER_NAME, FUNC_NAME) \
template<Target target, typename ImplEnum, ImplEnum impl, typename ... Ts> \
struct WRAPPER_NAME { \
	decltype(auto) operator() (Ts&& ... args) const { \
		return FUNC_NAME<target, impl>(std::forward<Ts>(args) ...); \
	} \
};


} // namespace Kokkidio

#endif
