set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
)

# WORKAROUND: clang-resource-headers is not in the exported targets
if(NOT TARGET clang-resource-headers)
  add_custom_target(clang-resource-headers)
endif()

add_clang_tool(mlir-clang
  mlir-clang.cc
  driver/cc1_main.cpp
  driver/cc1as_main.cpp
  driver/cc1gen_reproducer_main.cpp
  Lib/CGStmt.cc
  Lib/pragmaHandler.cc
  Lib/AffineUtils.cc
  Lib/ValueCategory.cc
  Lib/utils.cc
  Lib/IfScope.cc
)

target_include_directories(mlir-clang PRIVATE
  "${LLVM_SOURCE_DIR}/../clang/include"
  "${CMAKE_BINARY_DIR}/tools/clang/include"
)

target_compile_definitions(mlir-clang PUBLIC -DLLVM_OBJ_ROOT="${LLVM_BINARY_DIR}")
target_link_libraries(mlir-clang PRIVATE
  MLIRSCFTransforms
  MLIRPolygeist

  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRLLVMIR
  MLIRNVVMIR
  MLIROpenMP
  MLIRGPUOps
  MLIRTransforms
  MLIRSCFToStandard
  MLIRStandardToLLVM
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRTargetLLVMIRImport
  MLIRPolygeistTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
  MLIROpenMPToLLVMIRTranslation

  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangSerialization
)
add_dependencies(mlir-clang MLIRPolygeistOpsIncGen MLIRPolygeistPassIncGen)

add_custom_target(all-mlir-clang ALL DEPENDS mlir-clang)