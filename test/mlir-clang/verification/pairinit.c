// RUN: mlir-clang %s --function=func -S | FileCheck %s

struct pair {
    int x, y;
};

struct pair func() {
    struct pair tmp = {2, 3};
    return tmp;
}

// CHECK:   func @func(%arg0: memref<?x2xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x2xi32>
// CHECK-NEXT:     %1 = memref.subview %0[0, 0] [1, 2] [1, 1] : memref<1x2xi32> to memref<2xi32>
// CHECK-NEXT:     affine.store %c2_i32, %1[0] : memref<2xi32>
// CHECK-NEXT:     affine.store %c3_i32, %1[1] : memref<2xi32>
// CHECK-NEXT:     %2 = affine.load %0[0, 0] : memref<1x2xi32>
// CHECK-NEXT:     affine.store %2, %arg0[0, 0] : memref<?x2xi32>
// CHECK-NEXT:     %3 = affine.load %0[0, 1] : memref<1x2xi32>
// CHECK-NEXT:     affine.store %3, %arg0[0, 1] : memref<?x2xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
