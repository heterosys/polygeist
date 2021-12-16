// RUN: polygeist-opt --detect-reduction %s | FileCheck %s

module  {
  func @reduce_with_iter_args(%arg0: memref<?xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %1 = memref.alloca() : memref<1xf32>
    affine.store %cst, %1[0] : memref<1xf32>
    %2 = memref.alloca() : memref<1xf32>
    affine.store %cst_0, %2[0] : memref<1xf32>
    // CHECK:  %3 = affine.load %2[0] : memref<1xf32>
    // CHECK:  %4 = affine.load %1[0] : memref<1xf32>
    // CHECK:  %5:2 = affine.for %arg1 = 0 to %0 iter_args(%arg2 = %3, %arg3 = %4) -> (f32, f32) {
    // CHECK:    %9 = affine.load %arg0[%arg1] : memref<?xf32>
    // CHECK:    %10 = arith.addf %arg3, %9 : f32
    // CHECK:    %11 = arith.mulf %arg2, %9 : f32
    // CHECK:    affine.yield %11, %10 : f32, f32
    // CHECK:  }
    // affine.store %5#1, %1[0] : memref<1xf32>
    // affine.store %5#0, %2[0] : memref<1xf32>
    affine.for %arg1 = 0 to %0 {
      %6 = affine.load %2[0] : memref<1xf32>
      %7 = affine.load %1[0] : memref<1xf32>
      %8 = affine.load %arg0[%arg1] : memref<?xf32>
      %9 = arith.addf %7, %8 : f32
      %10 = arith.mulf %6, %8 : f32
      affine.store %9, %1[0] : memref<1xf32>
      affine.store %10, %2[0] : memref<1xf32>
    }
    %3 = affine.load %2[0] : memref<1xf32>
    %4 = affine.load %1[0] : memref<1xf32>
    %5 = arith.addf %4, %3 : f32
    return %5 : f32
  }
}
