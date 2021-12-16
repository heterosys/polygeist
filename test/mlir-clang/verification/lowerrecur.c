// RUN: mlir-clang %s --function=* -emit-llvm -S | FileCheck %s
// FIXME: LLVM dialect validation issue
// XFAIL: *

struct X{
 double* a;
 double* b;
 int c;
};

void perm(struct X* v) {
    v->a = v->b;
}

// CHECK: define void @perm
