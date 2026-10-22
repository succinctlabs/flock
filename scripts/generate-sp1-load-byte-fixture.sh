#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FLOCK_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
SP1_ROOT=${1:-"$FLOCK_ROOT/../sp1"}
EXPECTED_COMMIT=4f56564727906846f80b30532dde26eacfc7e44c
EXAMPLE_DIR="$SP1_ROOT/crates/core/machine/examples"
EXAMPLE="$EXAMPLE_DIR/flock_load_byte_oracle.rs"
CREATED_DIR=false

test -d "$SP1_ROOT/.git" || {
    echo "not an SP1 checkout: $SP1_ROOT" >&2
    exit 1
}
test "$(git -C "$SP1_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT" || {
    echo "SP1 must be at $EXPECTED_COMMIT" >&2
    exit 1
}
test -z "$(git -C "$SP1_ROOT" status --porcelain)" || {
    echo "SP1 checkout must be clean" >&2
    exit 1
}
test ! -e "$EXAMPLE" || {
    echo "refusing to overwrite $EXAMPLE" >&2
    exit 1
}

if test ! -d "$EXAMPLE_DIR"; then
    mkdir -p -- "$EXAMPLE_DIR"
    CREATED_DIR=true
fi

cleanup() {
    rm -f -- "$EXAMPLE"
    if test "$CREATED_DIR" = true; then
        rmdir -- "$EXAMPLE_DIR"
    fi
}
trap cleanup EXIT

cp -- "$SCRIPT_DIR/sp1-load-byte-oracle.rs" "$EXAMPLE"
(
    cd -- "$SP1_ROOT"
    RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable} \
        cargo run -p sp1-core-machine --example flock_load_byte_oracle
)
