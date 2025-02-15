// RUN: mlir-clang %s --function=* -S | FileCheck %s

extern "C" {

struct pair {
    int x, y;
};
void sub0(pair& a);
void sub(pair& a) {
    a.x++;
}

void kernel_deriche() {
    pair a;
    a.x = 32;;
    pair &b = a;
    sub0(b);
}

}

// CHECK:   func @sub(%arg0: memref<?x2xi32>)
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = affine.load %arg0[0, 0] : memref<?x2xi32>
// CHECK-NEXT:     %1 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:     affine.store %1, %arg0[0, 0] : memref<?x2xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func @kernel_deriche()
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x2xi32>
// CHECK-NEXT:     %1 = memref.cast %0 : memref<1x2xi32> to memref<?x2xi32>
// CHECK-NEXT:     %2 = memref.subview %0[0, 0] [1, 2] [1, 1] : memref<1x2xi32> to memref<2xi32>
// CHECK-NEXT:     affine.store %c32_i32, %2[0] : memref<2xi32>
// CHECK-NEXT:     call @sub0(%1) : (memref<?x2xi32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
