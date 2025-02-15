//===- utils.cc -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils.h"
#include "clang-mlir.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "clang/AST/Expr.h"

using namespace mlir;
using namespace llvm;
using namespace clang;

Operation *
mlirclang::replaceFuncByOperation(FuncOp f, StringRef opName, OpBuilder &b,
                                  SmallVectorImpl<mlir::Value> &input,
                                  SmallVectorImpl<mlir::Value> &output) {
  MLIRContext *ctx = f->getContext();
  assert(ctx->isOperationRegistered(opName) &&
         "Provided lower_to opName should be registered.");

  // NOTE: The attributes of the provided FuncOp is ignored.
  OperationState opState(b.getUnknownLoc(), opName, input,
                         f.getCallableResults(), {});
  opState.addOperands(output);
  auto op = b.createOperation(opState);
  return op;
}

// Adapted from clang/lib/CodeGen/CodeGenModule.cpp, keeping only the most
// common cases.
// TODO: add a public interface in clang
llvm::GlobalValue::LinkageTypes
mlirclang::getLLVMLinkage(clang::ASTContext &context, const clang::Decl *D) {
  GVALinkage Linkage;
  if (auto *VD = dyn_cast<VarDecl>(D))
    Linkage = context.GetGVALinkageForVariable(VD);
  else {
    Linkage = context.GetGVALinkageForFunction(cast<FunctionDecl>(D));
  }

  if (Linkage == GVA_Internal)
    return llvm::GlobalValue::LinkageTypes::InternalLinkage;
  if (D->hasAttr<WeakAttr>())
    return llvm::GlobalValue::LinkageTypes::WeakAnyLinkage;
  if (const auto *FD = D->getAsFunction())
    if (FD->isMultiVersion() && Linkage == GVA_AvailableExternally)
      return llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage;
  if (Linkage == GVA_AvailableExternally)
    return llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage;
  if (Linkage == GVA_DiscardableODR)
    return llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage;
  if (Linkage == GVA_StrongODR)
    return llvm::GlobalValue::LinkageTypes::WeakODRLinkage;
  if (D->hasAttr<SelectAnyAttr>())
    return llvm::GlobalValue::LinkageTypes::WeakODRLinkage;
  assert(Linkage == GVA_StrongExternal);
  return llvm::GlobalValue::LinkageTypes::ExternalLinkage;
}