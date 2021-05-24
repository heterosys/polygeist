#!/usr/bin/env bash
# This script build and install the Polygeist version that Polymer can work together with.

set -o errexit
set -o pipefail
set -o nounset

# Directory of this script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"
POLYGEIST_DIR="${DIR}/../../Polygeist-polymer"
POLYMER_DIR="${DIR}/../"

# Read the file that records the Polygeist git hash.
POLYGEIST_VERSION="$(cat "${DIR}/../polygeist-version.txt")"

echo ">>> Update and build Polygeist"
echo ""
echo "   The Polygeist version: ${POLYGEIST_VERSION}"
echo ""

echo ">>> Cloning and checkout Polygeist ..."

git clone https://github.com/wsmoses/Polygeist "${POLYGEIST_DIR}"
cd "${POLYGEIST_DIR}"
git checkout "${POLYGEIST_VERSION}"
cd - &>/dev/null

echo ">>> Linking Polymer to Polygeist ..."
rm -r "${POLYGEIST_DIR}/mlir/tools/polymer"
ln -s "${POLYMER_DIR}" "${POLYGEIST_DIR}/mlir/tools/polymer"

echo ">>> Building Polygeist ..."
cd "${POLYGEIST_DIR}"
mkdir build
cd build

# Comment out -G Ninja if you don't want to use that.
cmake ../llvm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang" \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DBUILD_POLYMER=ON \
  -DPLUTO_LIBCLANG_PREFIX="$(llvm-config --prefix)"

# Build
cmake --build . --target check-polymer

echo ">>> Done!"
echo ""
echo "    Polymer utilities are built under ${POLYGEIST_DIR}/build"
