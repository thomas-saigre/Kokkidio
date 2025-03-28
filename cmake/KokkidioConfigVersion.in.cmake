# kokkidio-config-version.cmake - checks version: major must match, minor must be less than or equal

set(PACKAGE_VERSION @Kokkidio_VERSION@)

if("${PACKAGE_FIND_VERSION_MAJOR}" EQUAL "@Kokkidio_VERSION_MAJOR@")
	if ("${PACKAGE_FIND_VERSION_MINOR}" EQUAL "@Kokkidio_VERSION_MINOR@")
		set(PACKAGE_VERSION_EXACT TRUE)
	elseif("${PACKAGE_FIND_VERSION_MINOR}" LESS "@Kokkidio_VERSION_MINOR@")
		set(PACKAGE_VERSION_COMPATIBLE TRUE)
	else()
		set(PACKAGE_VERSION_UNSUITABLE TRUE)
	endif()
else()
	set(PACKAGE_VERSION_UNSUITABLE TRUE)
endif()