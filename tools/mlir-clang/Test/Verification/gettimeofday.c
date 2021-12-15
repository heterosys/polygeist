// RUN: mlir-clang %s %stdinclude --function=alloc -S | FileCheck %s

#include <time.h>
#include <sys/time.h>
double alloc() {
  struct timeval Tp;
  gettimeofday(&Tp, NULL);
  return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

// CHECK:   func @alloc() -> f64
// CHECK-NEXT:     %cst = arith.constant 9.9999999999999995E-7 : f64
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr<struct<(i64, i64)>>
// CHECK-NEXT:     %1 = memref.alloca() : memref<1x2xi64>
// CHECK-NEXT:     %2 = llvm.mlir.null : !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT:     %3 = llvm.call @gettimeofday(%0, %2) : (!llvm.ptr<struct<(i64, i64)>>, !llvm.ptr<struct<(i32, i32)>>) -> i32
// CHECK-NEXT:     %4 = llvm.getelementptr %0[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:     %5 = llvm.load %4 : !llvm.ptr<i64>
// CHECK-NEXT:     affine.store %5, %1[0, 0] : memref<1x2xi64>
// CHECK-NEXT:     %6 = llvm.getelementptr %0[%c0_i32, %c1_i32] : (!llvm.ptr<struct<(i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr<i64>
// CHECK-NEXT:     affine.store %7, %1[0, 1] : memref<1x2xi64>
// CHECK-NEXT:     %8 = memref.subview %1[0, 0] [1, 2] [1, 1] : memref<1x2xi64> to memref<2xi64>
// CHECK-NEXT:     %9 = affine.load %8[0] : memref<2xi64>
// CHECK-NEXT:     %10 = arith.sitofp %9 : i64 to f64
// CHECK-NEXT:     %11 = affine.load %8[1] : memref<2xi64>
// CHECK-NEXT:     %12 = arith.sitofp %11 : i64 to f64
// CHECK-NEXT:     %13 = arith.mulf %12, %cst : f64
// CHECK-NEXT:     %14 = arith.addf %10, %13 : f64
// CHECK-NEXT:     return %14 : f64
// CHECK-NEXT:   }
