#!/bin/zsh
# abq — quick paired A/B on a MICRO-BENCH, for kernel-level work.
#
# Why: a full m=32 prove costs ~40 s (MT) or minutes (ST) per invocation, so a
# 3-pair A/B is 10-20 minutes. The zerocheck micro-benches drive the same
# kernels on a synthetic witness with no R1CS build, commit or open — the
# round-2 sweep runs m=16..29 in 0.6 s — and most of them print a CHECKSUM,
# which proves bit-identity for free. Validated 2026-08-31: the round-2 ports
# measured -16.2% on the full ST prove and -19.5% here, in 2 s instead of
# 20 minutes, with identical checksums.
#
# Usage:
#   benchmarks/abq.sh -b round2 -c /Users/buenz/flock-pre2 [-p 3] \
#                     [-e "RAYON_NUM_THREADS=1"] [-a "32 3"] [-g "(best)"]
#
#   -b  bench name (round1 | round2 | rounds3plus | ag_breakdown | ...)
#   -c  control worktree (built at the baseline commit)
#   -p  pairs (default 3)
#   -e  comma-separated env assignments applied to BOTH arms
#   -a  arguments passed to the bench binary
#   -g  grep pattern selecting the metric line (default "(best)")
#   -m  section to measure, for benches that sweep sizes (e.g. -m 29 selects
#       the "=== m = 29" block). Without it the FIRST matching line is used.
#
# Both arms are rebuilt first, then run in ALTERNATING order.
#
# LIMIT — READ THIS. The zerocheck micro-benches drive DENSE padding. The
# production union prove at m=32 takes the SPARSE round-2/tail dispatch, whose
# kernels run over short interval pieces with small, cache-resident outputs.
# Changes to per-pair ARITHMETIC transfer between the two (wideneon/qres read
# -19.5% here vs -16.2% on the real prove); changes to LOOP STRUCTURE do not —
# a two-pair unroll measured -3.5% here and +6.8% on the real ST prove
# (2026-09-01). Confirm anything structural with a full prove.
set -u

BENCH=""; CTL=""; PAIRS=3; ENVS=""; BARGS=""; PAT="(best)"; SECT=""
while getopts "b:c:p:e:a:g:m:" opt; do
  case $opt in
    b) BENCH=$OPTARG ;; c) CTL=$OPTARG ;; p) PAIRS=$OPTARG ;;
    e) ENVS=$OPTARG ;; a) BARGS=$OPTARG ;; g) PAT=$OPTARG ;; m) SECT=$OPTARG ;;
    *) echo "see header for usage" >&2; exit 2 ;;
  esac
done
[[ -n $BENCH && -n $CTL ]] || { echo "need -b and -c" >&2; exit 2 }

typeset -a E BA
E=(${(s:,:)ENVS})
BA=(${(z)BARGS})

# RELEASE profile deliberately, not `cargo bench`'s: the workspace's
# [profile.bench] adds lto="thin" + codegen-units=1, which costs 54 s to build
# against release's 20 s and measured the SAME number on the round-2 kernel
# (54.11 vs 53.7-55.5 ms at m=29 ST). Iterate under release; if a headline
# number is ever published, re-measure under `cargo bench`.
#
# Both arms build concurrently — the control is usually already current, but
# when it is not this halves the wall time.
echo "building both arms..." >&2
cargo build --release -p flock-prover --bench $BENCH 2>&1 | tail -1 >&2 &
pid_a=$!
(cd $CTL && cargo build --release -p flock-prover --bench $BENCH 2>&1 | tail -1 >&2) &
pid_b=$!
wait $pid_a; wait $pid_b

pick() { /bin/ls -t $1/target/release/deps/${BENCH}-* | grep -v '\.d$' | head -1 }
N=$(pick /Users/buenz/flock-main); C=$(pick $CTL)
echo "cand: $N" >&2; echo "ctrl: $C" >&2

# Metric = MIN over the matching lines INSIDE the -m section (section ends at
# the next "===" banner). Two traps this avoids, both of which bit:
#   * min across a whole sweep silently measures the SMALLEST size;
#   * the FIRST matching line is the COLD prove — for ag_breakdown that is
#     18.3 ms against a ~10 ms warm min, which manufactured a fake +15%
#     regression on 2026-09-01 before this was fixed.
sect() {
  if [[ -n $SECT ]]; then
    awk -v m="=== m = $SECT " 'index($0,m){p=1; next} p && /=== /{p=0} p'
  else cat; fi
}
run()  { env $E "$1" $BA 2>&1 | sect | grep -F "$PAT" | grep -oE "[0-9]+\.[0-9]+" | sort -n | head -1 }
sums() { env $E "$1" $BA 2>&1 | sect | grep -i checksum | head -1 }

typeset -a D
for (( i=1; i<=PAIRS; i++ )); do
  if (( i % 2 == 1 )); then a=$(run $N); b=$(run $C); else b=$(run $C); a=$(run $N); fi
  D+=($(python3 -c "print(f'{$a-$b:.3f}')"))
  echo "pair $i: cand=$a  ctrl=$b  delta=${D[-1]}"
done

python3 - "$@" <<PY
ds = [${(j:,:)D}]
ds.sort()
med = ds[len(ds)//2]
wins = sum(1 for x in ds if x < 0)
print(f"\nmedian delta {med:+.3f}   cand wins {wins}/{len(ds)}")
PY

sa=$(sums $N); sb=$(sums $C)
if [[ -n $sa || -n $sb ]]; then
  [[ $sa == $sb ]] && echo "checksums: IDENTICAL (bit-identity holds)" \
                   || { echo "checksums DIFFER:"; echo "  cand: $sa"; echo "  ctrl: $sb" }
fi
