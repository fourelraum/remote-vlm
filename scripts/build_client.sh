#!/usr/bin/env bash
# Configure & build the C++ client. Requires the system packages listed
# in README.md (libdatachannel, OpenCV, gRPC, protobuf, X11).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/client/build"

cmake -S "$ROOT/client" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "binary: $BUILD_DIR/remote_vlm_client"
