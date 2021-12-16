// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

#map = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK: func @main(%arg0: index) -> memref<30xi32, #map> {
// CHECK:   %0 = memref.alloca() : memref<30x30xi32>
// CHECK:   %1 = memref.subview %0[%arg0, 0] [1, 30] [1, 1] : memref<30x30xi32> to memref<30xi32, #map>
// CHECK:   return %1 : memref<30xi32, #map>
// CHECK: }
module {
  func @main(%arg0 : index) -> memref<30xi32, #map> {
    %0 = memref.alloca() : memref<30x30xi32>
    %1 = "polygeist.subindex"(%0, %arg0) : (memref<30x30xi32>, index) -> memref<30xi32, #map>
    return %1 : memref<30xi32, #map>
  }
}

// -----

#map = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK:  func @main(%arg0: index) -> memref<1000xi32, #map> {
// CHECK:    %0 = memref.alloca() : memref<2x1000xi32>
// CHECK:    %1 = memref.subview %0[%arg0, 0] [1, 1000] [1, 1] : memref<2x1000xi32> to memref<1000xi32, #map>
// CHECK:    return %1 : memref<1000xi32, #map>
// CHECK:  }
func @main(%arg0 : index) -> memref<1000xi32, #map> {
  %c0 = arith.constant 0 : index
  %1 = memref.alloca() : memref<2x1000xi32>
  %3 = "polygeist.subindex"(%1, %arg0) : (memref<2x1000xi32>, index) -> memref<?x1000xi32>
  %4 = "polygeist.subindex"(%3, %c0) : (memref<?x1000xi32>, index) -> memref<1000xi32, #map>
  return %4 : memref<1000xi32, #map>
}
