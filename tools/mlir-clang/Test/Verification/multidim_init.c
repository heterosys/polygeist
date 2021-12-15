// RUN: mlir-clang %s -S --function=foo | FileCheck %s

float foo(int i, int j) {
  // multiple dims with array fillers
  float A[][4] = {
      {1.0f, 2.0, 3.0, 4.0}, 
      {3.33333f},
      {0.1f, 0.2f, 0.3, 0.4},
  };

  // single dim
  float B[4] = {1.23f};

  float sum = 0.0f;
  // dynamic initialization
  for (int k = 0; k < 3; ++k) {
    float C[2] = {i + k, k - j};
    sum += C[i];
  }

  return A[i][j] + B[j] + sum;
}

// CHECK-LABEL: func @foo
// CHECK-DAG: %[[CST3:.*]] = arith.constant 3.33
// CHECK-DAG: %[[CST1_23:.*]] = arith.constant 1.23
// CHECK-DAG: %[[MEM_A:.*]] = memref.alloca() : memref<3x4xf32>
// CHECK-DAG: %[[MEM_B:.*]] = memref.alloca() : memref<4xf32>
// CHECK-DAG: %[[MEM_C:.*]] = memref.alloca() : memref<2xf32>
// CHECK: %[[SUBVIEW_A_0:.*]] = memref.subview %[[MEM_A]][0, 0] [1, 4] [1, 1] : memref<3x4xf32> to memref<4xf32>
// CHECK: affine.store %{{.*}}, %[[SUBVIEW_A_0]][0]
// CHECK: affine.store %{{.*}}, %[[SUBVIEW_A_0]][1]
// CHECK: affine.store %{{.*}}, %[[SUBVIEW_A_0]][2]
// CHECK: affine.store %{{.*}}, %[[SUBVIEW_A_0]][3]
// CHECK: %[[SUBVIEW_A_1:.*]] = memref.subview %[[MEM_A]][1, 0] [1, 4] [1, 1] : memref<3x4xf32> to memref<4xf32, #map0>
// CHECK: affine.store %[[CST3]], %[[SUBVIEW_A_1]][0]
// CHECK: affine.store %[[CST3]], %[[SUBVIEW_A_1]][1]
// CHECK: affine.store %[[CST3]], %[[SUBVIEW_A_1]][2]
// CHECK: affine.store %[[CST3]], %[[SUBVIEW_A_1]][3]
// CHECK: %[[SUBVIEW_A_2:.*]] = memref.subview %[[MEM_A]][2, 0] [1, 4] [1, 1] : memref<3x4xf32> to memref<4xf32, #map1>
// CHECK: affine.store %{{.*}}, %[[SUBVIEW_A_2]][0]
// CHECK: affine.store %{{.*}}, %[[SUBVIEW_A_2]][1]
// CHECK: affine.store %{{.*}}, %[[SUBVIEW_A_2]][2]
// CHECK: affine.store %{{.*}}, %[[SUBVIEW_A_2]][3]

// CHECK: affine.store %[[CST1_23]], %[[MEM_B]][0]
// CHECK: affine.store %[[CST1_23]], %[[MEM_B]][1]
// CHECK: affine.store %[[CST1_23]], %[[MEM_B]][2]
// CHECK: affine.store %[[CST1_23]], %[[MEM_B]][3]

// CHECK: scf.for
// CHECK: affine.store %{{.*}}, %[[MEM_C]][0]
// CHECK: affine.store %{{.*}}, %[[MEM_C]][1]
