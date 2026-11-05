#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
binary="${build_dir}/asan_smoke"

mkdir -p "${build_dir}"

clang++ \
    -std=c++17 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -fsanitize=address \
    -fno-omit-frame-pointer \
    -g \
    "${script_dir}/asan_smoke.cpp" \
    -o "${binary}"

ASAN_OPTIONS=halt_on_error=1 "${binary}"
