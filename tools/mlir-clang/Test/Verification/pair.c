// RUN: mlir-clang %s --function=* -S | FileCheck %s

typedef struct {
  int a, b;
} pair;

pair byval0(pair a, int x);
pair byval(pair a, int x) {
  a.b = x;
  return a;
}

int create() {
  pair p;
  p.a = 0;
  p.b = 1;
  pair p2 = byval0(p, 2);
  return p2.a;
}

// CHECK:   func @byval(%arg0: memref<?x2xi32>, %arg1: i32, %arg2: memref<?x2xi32>)
// CHECK-NEXT:     affine.store %arg1, %arg0[0, 1] : memref<?x2xi32>
// CHECK-NEXT:     %0 = affine.load %arg0[0, 0] : memref<?x2xi32>
// CHECK-NEXT:     affine.store %0, %arg2[0, 0] : memref<?x2xi32>
// CHECK-NEXT:     %1 = affine.load %arg0[0, 1] : memref<?x2xi32>
// CHECK-NEXT:     affine.store %1, %arg2[0, 1] : memref<?x2xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @create() -> i32
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x2xi32>
// CHECK-NEXT:     %1 = memref.alloca() : memref<1x2xi32>
// CHECK-NEXT:     %2 = memref.alloca() : memref<1x2xi32>
// CHECK-NEXT:     %3 = memref.alloca() : memref<1x2xi32>
// CHECK-NEXT:     %4 = memref.subview %3[0, 0] [1, 2] [1, 1] : memref<1x2xi32> to memref<2xi32>
// CHECK-NEXT:     affine.store %c0_i32, %4[0] : memref<2xi32>
// CHECK-NEXT:     affine.store %c1_i32, %4[1] : memref<2xi32>
// CHECK-NEXT:     %5 = affine.load %3[0, 0] : memref<1x2xi32>
// CHECK-NEXT:     affine.store %5, %2[0, 0] : memref<1x2xi32>
// CHECK-NEXT:     %6 = affine.load %3[0, 1] : memref<1x2xi32>
// CHECK-NEXT:     affine.store %6, %2[0, 1] : memref<1x2xi32>
// CHECK-NEXT:     %7 = memref.cast %2 : memref<1x2xi32> to memref<?x2xi32>
// CHECK-NEXT:     %8 = memref.cast %1 : memref<1x2xi32> to memref<?x2xi32>
// CHECK-NEXT:     call @byval0(%7, %c2_i32, %8) : (memref<?x2xi32>, i32, memref<?x2xi32>) -> ()
// CHECK-NEXT:     %9 = affine.load %1[0, 0] : memref<1x2xi32>
// CHECK-NEXT:     affine.store %9, %0[0, 0] : memref<1x2xi32>
// CHECK-NEXT:     %10 = affine.load %1[0, 1] : memref<1x2xi32>
// CHECK-NEXT:     affine.store %10, %0[0, 1] : memref<1x2xi32>
// CHECK-NEXT:     %11 = memref.subview %0[0, 0] [1, 2] [1, 1] : memref<1x2xi32> to memref<2xi32>
// CHECK-NEXT:     %12 = affine.load %11[0] : memref<2xi32>
// CHECK-NEXT:     return %12 : i32
// CHECK-NEXT:   }
