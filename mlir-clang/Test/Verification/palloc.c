// RUN: mlir-clang %s %stdinclude --function=init_array -S | FileCheck %s
// RUN: mlir-clang %s %stdinclude --function=init_array -S --memref-fullrank | FileCheck %s --check-prefix=FULLRANK

#include <stdio.h>
#include <unistd.h>
#include <string.h>

/* Include polybench common header. */
#include <polybench.h>

void use(double A[20]);
/* Array initialization. */

void init_array (int n)
{
  double (*B)[20] = (double(*)[20])polybench_alloc_data (20, sizeof(double)) ;
  (*B)[2] = 3.0;
  use(*B);
}


// CHECK:  func @init_array(%arg0: i32)
// CHECK-NEXT:    %cst = constant 3.000000e+00 : f64
// CHECK-NEXT:    %0 = memref.alloc() : memref<20xf64>
// CHECK-NEXT:    affine.store %cst, %0[2] : memref<20xf64>
// CHECK-NEXT:    %1 = memref.cast %0 : memref<20xf64> to memref<?xf64>
// CHECK-NEXT:    call @use(%1) : (memref<?xf64>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// FULLRANK: %[[VAL0:.*]] = memref.alloc() : memref<20xf64>
// FULLRANK: call @use(%[[VAL0]]) : (memref<20xf64>) -> ()