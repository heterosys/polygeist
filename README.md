# Polygeist

[![Build and Test](https://github.com/heterosys/polygeist/actions/workflows/build.yml/badge.svg)](https://github.com/heterosys/polygeist/actions/workflows/build.yml)

📥 C/C++ frontend to affine MLIR for polyhedral optimization.

## How to build Polygeist

### Requirements 
- Working C and C++ toolchains (`build-essential`)
- LLVM, Clang and MLIR 14.0.0 (development snapshot)
- Building and testing system: `cmake`, `lit` and `ninja`

### 0. Clone Polygeist and install prerequisites
```sh
git clone https://github.com/heterosys/polygeist.git
cd polygeist

sudo apt-get install -y build-essential python3-pip
pip3 install cmake ninja lit
```

### 1. Install LLVM, Clang, and MLIR

You can download our nightly pre-built snapshot from https://github.com/heterosys/llvm-nightly.

```sh
OS_DISTRIBUTER=$(lsb_release -is | tr '[:upper:]' '[:lower:]')
OS_RELEASE=$(lsb_release -rs)

LLVM_URL="https://github.com/heterosys/llvm-nightly/releases/latest/download/llvm-clang-mlir-dev-${OS_DISTRIBUTER}-${OS_RELEASE}.deb"

TEMP_DEB="$(mktemp)" && \
  wget -O "${TEMP_DEB}" ${LLVM_URL} && \
  (sudo dpkg -i "${TEMP_DEB}" || sudo apt-get -yf install)
rm -f "${TEMP_DEB}"
```

### 2. Build Polygeist

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_MAKE_PROGRAM=ninja -G Ninja \
  -DLLVM_EXTERNAL_LIT=`which lit`
cmake --build build --target all
```

To test `mlir-clang` and `polygeist-opt`:

```sh
cmake --build build --target check-mlir-clang
cmake --build build --target check-polygeist-opt
```

Cheers! 🍺
