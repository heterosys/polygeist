// RUN: mlir-clang %s --function=* -S | FileCheck %s
void reduction_gemm() {
  int i, j, k;
  int A[1024][1024];
  int B[1024][1024];
  int C[1024][1024];

#pragma scop
  for (i = 0; i < 1024; i++)
    for (j = 0; j < 1024; j++)
      for (k = 0; k < 1024; k++)
        C[i][j] += A[i][k] * B[k][j];
#pragma endscop
}
// CHECK:    affine.for %arg0 = 0 to 1024 {
// CHECK:      affine.for %arg1 = 0 to 1024 {
// CHECK:        affine.for %arg2 = 0 to 1024 {
// CHECK:          %3 = affine.load %2[%arg0, %arg2] : memref<1024x1024xi32>
// CHECK:          %4 = affine.load %1[%arg2, %arg1] : memref<1024x1024xi32>
// CHECK:          %5 = arith.muli %3, %4 : i32
// CHECK:          %6 = affine.load %0[%arg0, %arg1] : memref<1024x1024xi32>
// CHECK:          %7 = arith.addi %6, %5 : i32
// CHECK:          affine.store %7, %0[%arg0, %arg1] : memref<1024x1024xi32>

void reduction_bicg() {
  int i, j;
  int A[100][200];
  int r[100];
  int s[200];
  int p[200];
  int q[100];

#pragma scop
  for (i = 0; i < 100; i++) {
    for (j = 0; j < 200; j++) {
      s[j] = s[j] + r[i] * A[i][j];
    }
  }
#pragma endscop
}
// CHECK:    affine.for %arg0 = 0 to 100 {
// CHECK:      affine.for %arg1 = 0 to 200 {
// CHECK:        %3 = affine.load %0[%arg1] : memref<200xi32>
// CHECK:        %4 = affine.load %1[%arg0] : memref<100xi32>
// CHECK:        %5 = affine.load %2[%arg0, %arg1] : memref<100x200xi32>
// CHECK:        %6 = arith.muli %4, %5 : i32
// CHECK:        %7 = arith.addi %3, %6 : i32
// CHECK:        affine.store %7, %0[%arg1] : memref<200xi32>

void reduction_sum() {
  int sum = 0;
  int A[100];
#pragma scop
  for (int i = 0; i < 100; i++)
    sum += A[i];
#pragma endscop
}
// CHECK:    affine.for %arg0 = 0 to 100 {
// CHECK:      %2 = affine.load %0[%arg0] : memref<100xi32>
// CHECK:      %3 = affine.load %1[0] : memref<1xi32>
// CHECK:      %4 = arith.addi %3, %2 : i32
// CHECK:      affine.store %4, %1[0] : memref<1xi32>
