set(MLIR_CLANG_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_CLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../)
if(EXISTS ${LLVM_BUILD_MAIN_SRC_DIR}/../clang/lib/Headers)
  set(CLANG_HEADER_DIR ${LLVM_BUILD_MAIN_SRC_DIR}/../clang/lib/Headers)
else()
  set(CLANG_VERSION_STRING ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH})
  set(CLANG_HEADER_DIR /usr/lib/clang/${CLANG_VERSION_STRING}/include)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg
)

list(APPEND MLIR_CLANG_TEST_DEPS
  llvm-config 
  mlir-clang
  split-file
  clang
  )

add_lit_testsuite(check-mlir-clang "Running the clang-to-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} 
  DEPENDS ${MLIR_CLANG_TEST_DEPS}
  )

add_lit_testsuite(check-mlir-clang-single "Running the clang-to-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} 
  DEPENDS ${MLIR_CLANG_TEST_DEPS}
  ARGS -j 1
  )

set_target_properties(check-mlir-clang PROPERTIES FOLDER "clang-to-mlir tests")