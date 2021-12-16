cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(polygeist LANGUAGES CXX C)

  set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

  find_package(MLIR REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
else ()
  set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIRS ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(POLYGEIST_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/polygeist/include)
  include_directories(${MLIR_TABLEGEN_OUTPUT_DIR})
  include_directories(${POLYGEIST_TABLEGEN_OUTPUT_DIR})
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddClang)
include(HandleLLVMOptions)

set(POLYGEIST_TOOLS_DIR ${CMAKE_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(POLYGEIST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(POLYGEIST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(POLYGEIST_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include )
set(POLYGEIST_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(POLYGEIST_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(sanitizers)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

include(GNUInstallDirs)
install(
  TARGETS polygeist-opt
  EXPORT PolygeistTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(
  TARGETS mlir-clang
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/include/polygeist"
  DIRECTORY "${CMAKE_BINARY_DIR}/include/polygeist"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "CMakeFiles" EXCLUDE)

set_target_properties(polygeist-opt PROPERTIES EXPORT_NAME polygeist)
export(
  EXPORT PolygeistTargets
  NAMESPACE polygeist::
  FILE ${CMAKE_CURRENT_BINARY_DIR}/polygeist/PolygeistTargets.cmake)
set(ConfigPackageLocation lib/cmake/polygeist)
install(
  EXPORT PolygeistTargets
  FILE PolygeistTargets.cmake
  NAMESPACE polygeist::
  DESTINATION ${ConfigPackageLocation})

set(CPACK_PACKAGE_NAME polygeist)
if(NOT DEFINED CPACK_PACKAGE_VERSION_MAJOR)
  set(CPACK_PACKAGE_VERSION_MAJOR 0)
endif()
if(NOT DEFINED CPACK_PACKAGE_VERSION_MINOR)
  set(CPACK_PACKAGE_VERSION_MINOR 0)
endif()
if(NOT DEFINED CPACK_PACKAGE_VERSION_PATCH)
  set(CPACK_PACKAGE_VERSION_PATCH git)
endif()
if(NOT DEFINED PACKAGE_VERSION)
  set(PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
endif()

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C/C++ frontend to affine MLIR")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jason Lau")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
set(CPACK_PACKAGE_VERSION ${PACKAGE_VERSION})
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

set(CPACK_DEBIAN_PACKAGE_DEPENDS "zlib1g-dev,libncurses5-dev")
set(CPACK_GENERATOR DEB)
include(CPack)

add_subdirectory(test)
