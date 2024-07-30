# checks whether VAR_TO_CHECK is defined as either an environment or CMake variable,
# and if either is the case, it sets VAR_TO_SET to that value.
macro(set_if_defined VAR_TO_SET VAR_TO_CHECK)

	# Check if VAR_TO_CHECK is defined as a CMake variable.
	if (DEFINED ${VAR_TO_CHECK})
		set(${VAR_TO_SET} ${${VAR_TO_CHECK}})
		message(STATUS "${VAR_TO_CHECK}=${${VAR_TO_SET}} (found as a CMake variable)")
	endif()

	# Check if VAR_TO_CHECK is defined as an environment variable.
	if (DEFINED ENV{${VAR_TO_CHECK}})
		if (DEFINED ${VAR_TO_SET})
			message(STATUS
				"${VAR_TO_CHECK} found as both CMake AND environment variable. "
				"CMake variable takes precedence."
			)
		else()
			string(REPLACE ":" ";" ${VAR_TO_SET}  $ENV{${VAR_TO_CHECK}})
			# set(${VAR_TO_SET} $ENV{${VAR_TO_CHECK}})
			message(STATUS "${VAR_TO_CHECK}=${${VAR_TO_SET}} (found as a environment variable)")
		endif()
	endif()

	# If VAR_TO_SET was set, print its value.
	if(NOT DEFINED ${VAR_TO_SET})
		message(STATUS "${VAR_TO_CHECK} was not found as either environment or CMake variable.")
	endif()
endmacro()