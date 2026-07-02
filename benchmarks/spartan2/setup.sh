#!/usr/bin/env bash
# setup.sh — clone microsoft/Spartan2 and run its SHA-256 benchmarks.
#
# This is the only checked-in file under spartan2/; everything else (the cloned
# repo) is gitignored. It is self-contained so it works from a fresh checkout
# where spartan2/ contains nothing but this script.
#
# The clone is pinned to the same commit the benchmarks/ harness
# measured (override with SPARTAN2_SHA, or SPARTAN2_SHA="" for upstream HEAD)
# so numbers stay comparable across both harnesses.
#
# Two criterion benches; select with SPARTAN_BENCH (default: all):
#   spartan      — `sha256_spartan`: proves SHA-256 of a 1 KiB / 2 KiB preimage
#                  with the T256HyraxEngine (curve-based 256-bit field, Hyrax
#                  PCS). 1 KiB = ceil((1024+9)/64) = 17 sequentially-dependent
#                  compressions; 2 KiB = 33.
#   neutronnova  — `sha256_neutronnova`: NeutronNova folds N SHA-256 step
#                  circuits into one Spartan proof (1 KiB → 16 steps, 2 KiB → 32).
#   all          — both (default).
#   none         — clone/build only, run nothing (CI / setup-only).
#
# Closest Flock comparison (sequentially-dependent SHA-256 compressions):
#   cargo bench --bench sha2_chain_proof
#
# Thread count: BENCH_THREADS is a comma-separated sweep list consumed by
# Spartan2's bench harness (e.g. "1,4"). Defaults to the performance-core
# count, mirroring Flock's init_perf_thread_pool so the headline numbers are
# compared at the same thread count; set BENCH_THREADS to override.
#
# SPARTAN_FILTER (optional) is passed through as criterion's benchmark-id
# regex, e.g. '^spartan_sha256/(prove|verify)/' to skip the slow setup /
# prep_prove groups. Used by ../bench_sha256.sh.
#
# Usage:
#   ./spartan2/setup.sh                          # clone (if needed) + run both
#   SPARTAN_BENCH=spartan ./spartan2/setup.sh    # only the Hyrax/Spartan bench
#   BENCH_THREADS=1,4 ./spartan2/setup.sh        # ST + 4-thread sweep
#   SPARTAN_BENCH=none ./spartan2/setup.sh       # fetch + compile only

set -euo pipefail

REPO_URL="https://github.com/microsoft/Spartan2.git"
# Default pin = the commit benchmarks/setup.sh measures.
SPARTAN2_SHA="${SPARTAN2_SHA-0d4f1409e8f30536b8b25ed3f81bc446ed717e61}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clone the repo into DIR (alongside this script) if it isn't already there.
# Clone into a temp dir first, since DIR is non-empty (it holds this script).
if [[ ! -f "$DIR/Cargo.toml" ]]; then
	echo "Cloning $REPO_URL into $DIR ..."
	TMP="$(mktemp -d)"
	trap 'rm -rf "$TMP"' EXIT
	git clone "$REPO_URL" "$TMP/spartan2"
	if [[ -n "$SPARTAN2_SHA" ]]; then
		git -C "$TMP/spartan2" checkout --quiet "$SPARTAN2_SHA"
		echo "Pinned to $SPARTAN2_SHA."
	fi
	shopt -s dotglob          # include dotfiles (.cargo, .github, ...)
	mv "$TMP/spartan2"/* "$DIR"/
	shopt -u dotglob
	# Drop the nested .git: a git repo inside spartan2/ would make the outer
	# flock-rust repo treat spartan2/ as an embedded repo and refuse to track
	# this script. It isn't needed to build/benchmark.
	rm -rf "$DIR/.git"
else
	echo "spartan2 already present in $DIR, skipping clone."
fi

# Apply the size-override patches (idempotent). Stock sha256_neutronnova and
# sha256_spartan hardcode their compression counts; these patches add NN_BYTES /
# SP_BYTES env knobs so ../bench_sha256.sh can sweep batch sizes (NeutronNova up
# to its throughput peak; Spartan-Hyrax up to where its monolithic cost makes
# throughput decline). The clone's .git was stripped above, so apply with
# `patch`, not `git apply`. Sentinel for each: the env-var name in the bench.
for spec in "neutronnova_nn_bytes.patch:sha256_neutronnova.rs:NN_BYTES" \
            "spartan_sp_bytes.patch:sha256_spartan.rs:SP_BYTES"; do
	pf="${spec%%:*}"; rest="${spec#*:}"; bench="${rest%%:*}"; sentinel="${rest##*:}"
	[[ -f "$DIR/$pf" ]] || continue
	if grep -q "$sentinel" "$DIR/benches/$bench" 2>/dev/null; then
		echo "$sentinel patch already applied, skipping."
	else
		echo "Applying $sentinel size-override patch ..."
		patch -p1 -d "$DIR" < "$DIR/$pf"
	fi
done

# Thread sweep for Spartan2's bench harness. Default: P-core count only, like
# the other benchmarks setups (binius64, plonky3) and Flock itself.
if [[ -n "${BENCH_THREADS:-}" ]]; then
	THREADS_NOTE="user override"
else
	if [[ "$(uname -s)" == "Darwin" ]]; then
		# hw.perflevel0.physicalcpu = P-core count on Apple silicon.
		PCORES="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)"
	fi
	# Fallback (non-macOS, or sysctl unavailable): all logical CPUs.
	: "${PCORES:=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
	BENCH_THREADS="$PCORES"
	THREADS_NOTE="performance cores; matches Flock"
fi
export BENCH_THREADS

echo "=== BENCH_THREADS=$BENCH_THREADS ($THREADS_NOTE) ==="

BENCH="${SPARTAN_BENCH:-all}"

run_spartan() {
	echo "=== sha256_spartan: SHA-256 of 1/2 KiB preimage, T256HyraxEngine (cf. Flock sha2_chain_proof) ==="
	RUSTFLAGS="-C target-cpu=native" BENCH_THREADS="$BENCH_THREADS" \
		cargo bench --manifest-path "$DIR/Cargo.toml" --bench sha256_spartan \
		${SPARTAN_FILTER:+-- "$SPARTAN_FILTER"}
}

run_neutronnova() {
	echo "=== sha256_neutronnova: NeutronNova-folded SHA-256 step circuits ==="
	RUSTFLAGS="-C target-cpu=native" BENCH_THREADS="$BENCH_THREADS" \
		cargo bench --manifest-path "$DIR/Cargo.toml" --bench sha256_neutronnova \
		${SPARTAN_FILTER:+-- "$SPARTAN_FILTER"}
}

case "$BENCH" in
	spartan)     run_spartan ;;
	neutronnova) run_neutronnova ;;
	all)         run_spartan; echo; run_neutronnova ;;
	none)
		echo "SPARTAN_BENCH=none — compiling benches without running."
		RUSTFLAGS="-C target-cpu=native" \
			cargo bench --manifest-path "$DIR/Cargo.toml" --no-run
		;;
	*) echo "unknown SPARTAN_BENCH='$BENCH' (expected: spartan | neutronnova | all | none)" >&2; exit 1 ;;
esac
