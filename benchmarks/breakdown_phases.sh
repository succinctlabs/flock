#!/usr/bin/env bash
# breakdown_phases.sh — per-phase prover breakdown at a fixed BLAKE3 batch,
# split the way the RS-path porting work needs it:
#
#   witness | commit | zc round 1 | zc round 2 | zc rounds 3+ | lincheck | open
#
# breakdown.sh reports five coarse phases across three hashes at 2^14; this
# script instead pins one hash (BLAKE3) at one size (default 2^18, the ranked
# contract's size) and splits the zerocheck into round 1 (the univariate-skip
# URM, a full-size pass) vs rounds 2+ (fused fold + multilinear tail), because
# those two are optimized independently and need separate attribution.
#
# Sources:
#   - `FLOCK_PHASE_TSV=1` makes blake3_proof emit full-precision `PHASE` rows
#     (the human-readable rows round seconds to 2 decimals, ~+/-5 ms on a 1 s
#     phase, too coarse for small wins).
#   - `FLOCK_ZC_TIMING=1` makes zerocheck::prove_packed trace its sub-phases on
#     stderr. The LAST triple belongs to the prove_fast_timed breakdown pass
#     (earlier triples are the warm-up and the timed headline runs).
#
# Knobs:
#   TARGET_LOG2  log2 BLAKE3 compression count (default 18)
#   MT_THREADS   MT pass thread count (default: physical P-core count)
#   LABEL        tag for this run, recorded in the TSV (default: git HEAD short)
#   OUT          append machine-readable rows here (default: none)
#
# Usage:
#   ./benchmarks/breakdown_phases.sh
#   LABEL=baseline OUT=bd.tsv ./benchmarks/breakdown_phases.sh
#   TARGET_LOG2=16 ./benchmarks/breakdown_phases.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

H="${TARGET_LOG2:-18}"
LABEL="${LABEL:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"
OUT="${OUT:-}"
if [[ -z "${MT_THREADS:-}" ]]; then
	MT_THREADS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)"
	[[ -n "$MT_THREADS" ]] || MT_THREADS="$(getconf _NPROCESSORS_ONLN)"
fi
[[ "$H" =~ ^[0-9]+$ && "$H" -ge 1 ]] || { echo "TARGET_LOG2 must be a positive integer" >&2; exit 1; }
[[ "$MT_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "MT_THREADS must be a positive integer" >&2; exit 1; }

N=$(( 1 << H ))
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "=== Flock RS-path phase breakdown — BLAKE3 2^$H = $N compressions ==="
echo "label: $LABEL   headline: best-of-3 (fixed by the bench)   threads: 1 and $MT_THREADS"
echo

# -p flock-prover keeps Cargo from unifying features across the workspace (a
# bare build can pull in a global allocator that collides with this bench's own).
echo "building blake3_proof ..." >&2
cargo build --release -p flock-prover --bench blake3_proof >/dev/null 2>"$work/build.log" || {
	echo "build failed:" >&2; cat "$work/build.log" >&2; exit 1; }

for T in 1 "$MT_THREADS"; do
	echo "running ${T}-thread pass ..." >&2
	BLAKE3_LOG2S="$H" FLOCK_PHASE_TSV=1 FLOCK_ZC_TIMING=1 \
		RAYON_NUM_THREADS="$T" \
		cargo bench -p flock-prover --bench blake3_proof \
		>"$work/t$T.out" 2>"$work/t$T.err" || {
			echo "bench failed at T=$T:" >&2; tail -20 "$work/t$T.err" >&2; exit 1; }
done

# Pull one phase (seconds) from the PHASE rows.
phase_s() { awk -F '\t' -v k="$2" '$1=="PHASE" && $3==k {v=$4} END{if(v=="")exit 1; print v}' "$1"; }
# Pull the last [zc-timing] value (ms) whose label matches.
zc_ms() { grep -F "[zc-timing] $2:" "$1" | tail -1 | sed -E 's/.*: *([0-9.]+) ms.*/\1/'; }
comp_s() { sed -E -n 's/.*\(([0-9]+) compressions\/sec\).*/\1/p' "$1" | tail -1; }

emit() {
	local f_out="$1" f_err="$2" threads="$3"
	local witness commit zc lincheck open r1 r2 tl total head_cps
	# `|| true` so a parse miss reports below instead of tripping set -e silently.
	witness=$(phase_s "$f_out" witness || true); commit=$(phase_s "$f_out" commit || true)
	zc=$(phase_s "$f_out" zerocheck || true);    lincheck=$(phase_s "$f_out" lincheck || true)
	open=$(phase_s "$f_out" open || true)
	r1=$(zc_ms "$f_err" "round1 URM" || true)
	r2=$(zc_ms "$f_err" "round2 fused fold" || true)
	tl=$(zc_ms "$f_err" "rounds 3+ tail" || true)
	head_cps=$(comp_s "$f_out" || true)
	for v in witness commit zc lincheck open r1 r2 tl; do
		[[ -n "${!v}" ]] || { echo "breakdown_phases: could not parse '$v' at T=$threads" >&2
			echo "  stdout: $f_out   stderr: $f_err" >&2; exit 1; }
	done

	awk -v w="$witness" -v c="$commit" -v zc="$zc" -v lc="$lincheck" -v op="$open" \
	    -v r1="$r1" -v r2="$r2" -v tl="$tl" -v T="$threads" -v L="$LABEL" \
	    -v N="$N" -v cps="$head_cps" -v out="$OUT" '
	BEGIN {
		w*=1000; c*=1000; zc*=1000; lc*=1000; op*=1000
		total = w + c + r1 + r2 + tl + lc + op
		printf "  %-26s %10s %8s\n", "phase (" T "T)", "ms", "%"
		printf "  %-26s %10s %8s\n", "--------------------------", "----------", "-------"
		split("witness:" w " commit:" c " zc_round1:" r1 " zc_round2:" r2 " zc_rounds3plus:" tl " lincheck:" lc " open:" op, rows, " ")
		for (i = 1; i <= 7; i++) {
			split(rows[i], kv, ":")
			printf "  %-26s %10.2f %7.1f%%\n", kv[1], kv[2], 100 * kv[2] / total
			if (out != "") printf "%s\t%s\t%s\t%s\t%.4f\n", L, N, T, kv[1], kv[2] >> out
		}
		printf "  %-26s %10.2f %7.1f%%\n", "TOTAL", total, 100
		printf "\n  reconcile: zc round1+2+3plus = %.2f ms vs coarse zerocheck %.2f ms (delta %.2f)\n", r1 + r2 + tl, zc, r1 + r2 + tl - zc
		if (cps != "") printf "  headline: %s compressions/sec\n", cps
		if (out != "") printf "%s\t%s\t%s\t%s\t%.4f\n", L, N, T, "TOTAL", total >> out
	}'
	echo
}

emit "$work/t1.out" "$work/t1.err" 1
emit "$work/t${MT_THREADS}.out" "$work/t${MT_THREADS}.err" "$MT_THREADS"
[[ -n "$OUT" ]] && echo "appended machine-readable rows to $OUT"
