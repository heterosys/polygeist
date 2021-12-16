// RUN: mlir-clang %s %stdinclude --function=alloc -S | FileCheck %s

#include <time.h>
#include <sys/time.h>
double alloc() {
  struct timeval Tp;
  gettimeofday(&Tp, NULL);
  return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

// CHECK:   func @alloc() -> f64
// CHECK:     %cst = arith.constant 9.9999999999999995E-7 : f64
// CHECK:     %c1_i32 = arith.constant 1 : i32
// CHECK:     %c0_i32 = arith.constant 0 : i32
// CHECK:     %c1_i64 = arith.constant 1 : i64
// CHECK:     %[[ALLOC:.*]] = llvm.alloca %c1_i64 x !llvm.struct<(i64, i64)>
// CHECK:     %[[VAL:.*]] = memref.alloca()
// CHECK:     %[[NULL:.*]] = llvm.mlir.null
// CHECK:     %{{.*}} = llvm.call @gettimeofday(%[[ALLOC]], %[[NULL]])
// CHECK:     %[[D4:.*]] = llvm.getelementptr %[[ALLOC]][%c0_i32, %c0_i32]
// CHECK:     %[[D5:.*]] = llvm.load %[[D4]]
// CHECK:     affine.store %[[D5]], %[[VAL]][0, 0]
// CHECK:     %[[D6:.*]] = llvm.getelementptr %[[ALLOC]][%c0_i32, %c1_i32]
// CHECK:     %[[D7:.*]] = llvm.load %[[D6]]
// CHECK:     affine.store %[[D7]], %[[VAL]][0, 1]
// CHECK:     %[[SUBVIEW:.*]] = memref.subview %[[VAL]][0, 0] [1, 2] [1, 1]
// CHECK:     %[[SUBVIEW0:.*]] = affine.load %[[SUBVIEW]][0]
// CHECK:     %[[SUBVIEW0F:.*]] = arith.sitofp %[[SUBVIEW0]]
// CHECK:     %[[SUBVIEW1:.*]] = affine.load %[[SUBVIEW]][1]
// CHECK:     %[[SUBVIEW1F:.*]] = arith.sitofp %[[SUBVIEW1]]
// CHECK:     %[[D13:.*]] = arith.mulf %[[SUBVIEW1F]], %cst
// CHECK:     %[[RET:.*]] = arith.addf %[[SUBVIEW0F]], %[[D13]]
// CHECK:     return %[[RET]] : f64
// CHECK:   }
