// RUN: mlir-clang %s --function=* -S | FileCheck %s

typedef struct {
  int a, b;
} pair;

pair byval0(pair* a, int x);
pair byval(pair* a, int x) {
  return *a;
}

int create() {
  pair p;
  p.a = 0;
  p.b = 1;
  pair p2 = byval0(&p, 2);
  return p2.a;
}

// CHECK:   func @byval(%arg0: memref<?x2xi32>, %arg1: i32, %arg2: memref<?x2xi32>)
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
// CHECK-NEXT:     %3 = memref.subview %2[0, 0] [1, 2] [1, 1] : memref<1x2xi32> to memref<2xi32>
// CHECK-NEXT:     affine.store %c0_i32, %3[0] : memref<2xi32>
// CHECK-NEXT:     affine.store %c1_i32, %3[1] : memref<2xi32>
// CHECK-NEXT:     %4 = memref.cast %2 : memref<1x2xi32> to memref<?x2xi32>
// CHECK-NEXT:     %5 = memref.cast %1 : memref<1x2xi32> to memref<?x2xi32>
// CHECK-NEXT:     call @byval0(%4, %c2_i32, %5) : (memref<?x2xi32>, i32, memref<?x2xi32>) -> ()
// CHECK-NEXT:     %6 = affine.load %1[0, 0] : memref<1x2xi32>
// CHECK-NEXT:     affine.store %6, %0[0, 0] : memref<1x2xi32>
// CHECK-NEXT:     %7 = affine.load %1[0, 1] : memref<1x2xi32>
// CHECK-NEXT:     affine.store %7, %0[0, 1] : memref<1x2xi32>
// CHECK-NEXT:     %8 = memref.subview %0[0, 0] [1, 2] [1, 1] : memref<1x2xi32> to memref<2xi32>
// CHECK-NEXT:     %9 = affine.load %8[0] : memref<2xi32>
// CHECK-NEXT:     return %9 : i32
