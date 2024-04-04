
# GLSL files
# FILE(GLOB GLSL "*.glsl")
# Main source files
FILE(GLOB_RECURSE MAIN_SOURCES "*.cpp")
FILE(GLOB_RECURSE MAIN_HEADERS "*.h")
# Evaluator source files
FILE(GLOB EVAL_CUDA "*.cu")
FILE(GLOB EVAL_CUDA_H "*.cuh")
FILE(GLOB EVAL_HEADERS "*.h")

# Set the executable.
ADD_EXECUTABLE(DAO
	${MAIN_SOURCES} ${MAIN_HEADERS}# ${GLSL}
	${EVAL_CUDA} ${EVAL_CUDA_H} ${EVAL_HEADERS}
)

set(CMAKE_CUDA_ARCHITECTURES "native")
enable_language("CUDA")

TARGET_LINK_LIBRARIES(DAO
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
)

# LINK IMGUI
#TARGET_LINK_LIBRARIES(DAO
#	imgui
#)

# Get the GLM environment variable. Since GLM is a header-only library, we
# just need to add it to the include directory.
SET(GLM_INCLUDE_DIR "$ENV{GLM_INCLUDE_DIR}")
IF(NOT GLM_INCLUDE_DIR)
	# The environment variable was not set
	SET(ERR_MSG "Please point the environment variable GLM_INCLUDE_DIR to the root directory of your GLM installation.")
	IF(WIN32)
		# On Windows, try the default location
		MESSAGE(STATUS "Looking for GLM in ${DEF_DIR_GLM}")
		IF(IS_DIRECTORY ${DEF_DIR_GLM})
			MESSAGE(STATUS "Found!")
			SET(GLM_INCLUDE_DIR ${DEF_DIR_GLM})
		ELSE()
			MESSAGE(FATAL_ERROR ${ERR_MSG})
		ENDIF()
	ELSE()
		MESSAGE(FATAL_ERROR ${ERR_MSG})
	ENDIF()
ENDIF()
INCLUDE_DIRECTORIES(${GLM_INCLUDE_DIR})

# Get the GLEW environment variable.
SET(GLEW_DIR "$ENV{GLEW_DIR}")
IF(NOT GLEW_DIR)
	# The environment variable was not set
	SET(ERR_MSG "Please point the environment variable GLEW_DIR to the root directory of your GLEW installation.")
	IF(WIN32)
		# On Windows, try the default location
		MESSAGE(STATUS "Looking for GLEW in ${DEF_DIR_GLEW}")
		IF(IS_DIRECTORY ${DEF_DIR_GLEW})
			MESSAGE(STATUS "Found!")
			SET(GLEW_DIR ${DEF_DIR_GLEW})
		ELSE()
			MESSAGE(FATAL_ERROR ${ERR_MSG})
		ENDIF()
	ELSE()
		MESSAGE(FATAL_ERROR ${ERR_MSG})
	ENDIF()
ENDIF()
INCLUDE_DIRECTORIES(${GLEW_DIR})

# Get the GLFW environment variable. There should be a CMakeLists.txt in the 
# specified directory.
SET(GLFW_DIR "$ENV{GLFW_DIR}")
IF(NOT GLFW_DIR)
	# The environment variable was not set
	SET(ERR_MSG "Please point the environment variable GLFW_DIR to the root directory of your GLFW installation.")
	IF(WIN32)
		# On Windows, try the default location
		MESSAGE(STATUS "Looking for GLFW in ${DEF_DIR_GLFW}")
		IF(IS_DIRECTORY ${DEF_DIR_GLFW})
			MESSAGE(STATUS "Found!")
			SET(GLFW_DIR ${DEF_DIR_GLFW})
		ELSE()
			MESSAGE(FATAL_ERROR ${ERR_MSG})
		ENDIF()
	ELSE()
		MESSAGE(FATAL_ERROR ${ERR_MSG})
	ENDIF()
ENDIF()
OPTION(GLFW_BUILD_EXAMPLES "GLFW_BUILD_EXAMPLES" OFF)
OPTION(GLFW_BUILD_TESTS "GLFW_BUILD_TESTS" OFF)
OPTION(GLFW_BUILD_DOCS "GLFW_BUILD_DOCS" OFF)
IF(CMAKE_BUILD_TYPE MATCHES Release)
	ADD_SUBDIRECTORY(${GLFW_DIR} ${GLFW_DIR}/release)
ELSE()
	ADD_SUBDIRECTORY(${GLFW_DIR} ${GLFW_DIR}/debug)
ENDIF()
INCLUDE_DIRECTORIES(${GLFW_DIR}/include)
TARGET_LINK_LIBRARIES(DAO glfw ${GLFW_LIBRARIES})


INCLUDE_DIRECTORIES(${GLEW_DIR}/include)
IF(WIN32)
	# With prebuilt binaries
	# Check for 32 vs 64 bit generator
	IF(NOT CMAKE_CL_64)
		MESSAGE(STATUS "Using 32Bit")
		TARGET_LINK_LIBRARIES(DAO ${GLEW_DIR}/lib/Release/Win32/glew32s.lib)
	ELSE()
		MESSAGE(STATUS "Using 64Bit")
		TARGET_LINK_LIBRARIES(DAO ${GLEW_DIR}/lib/Release/x64/glew32s.lib)
	ENDIF()
ELSE()
	TARGET_LINK_LIBRARIES(DAO ${GLEW_DIR}/lib/libGLEW.a)
ENDIF()

# Set EIGEN
SET(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}")
IF(NOT EIGEN3_INCLUDE_DIR)
	# The environment variable was not set
	SET(ERR_MSG "EIGEN not found.")
	IF(WIN32)
		# On Windows, try the default location
		MESSAGE(STATUS "Looking for EIGEN in ${DEF_DIR_EIGEN}")
		IF(IS_DIRECTORY ${DEF_DIR_EIGEN})
			MESSAGE(STATUS "Found!")
			SET(EIGEN3_INCLUDE_DIR ${DEF_DIR_EIGEN})
		ELSE()
			MESSAGE(FATAL_ERROR ${ERR_MSG})
		ENDIF()
	ELSE()
		MESSAGE(FATAL_ERROR ${ERR_MSG})
	ENDIF()
ENDIF()
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

# # Use c++20
SET_TARGET_PROPERTIES(DAO PROPERTIES CXX_STANDARD 20)
SET_TARGET_PROPERTIES(DAO PROPERTIES LINKER_LANGUAGE CXX)

# OS specific options and libraries
IF(WIN32)
	# -Wall produces way too many warnings.
	# -pedantic is not supported.
	# Disable warning 4996.
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")
	TARGET_LINK_LIBRARIES(DAO opengl32.lib)
	SET_PROPERTY(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT DAO)
ELSE()
	# Enable all pedantic warnings.
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
	IF(APPLE)
		# Add required frameworks for GLFW.
		TARGET_LINK_LIBRARIES(DAO "-framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo")
	ELSE()
		#Link the Linux OpenGL library
		TARGET_LINK_LIBRARIES(DAO "GL")
	ENDIF()
ENDIF()