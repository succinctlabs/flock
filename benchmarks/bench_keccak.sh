#!/usr/bin/env bash
# bench_keccak.sh — run Flock + the competitor keccak benchmarks and print a
# combined summary table (throughput, prove time, proof size).
#
# - flock:       `cargo bench --bench keccak3_proof` (KECCAK3_KS mode) — Flock's
#                3-wide keccak encoder (~97% block utilization vs a naive ~65%);
#                prove_fast = full fast prover, incl. witness. Proves 3·2^(h-1)
#                keccaks per 2^h bucket (= 1.5x the 2^h target), where the 3-wide
#                packing wins; the "keccaks" column shows the actual proven count.
# - binius64:    benchmarks/binius64/setup.sh (independent permutations)
# - plonky3:     benchmarks/plonky3/setup.sh  (real Plonky3 example)
# - hashcaster:  from ./bench-hash-in-snark via its bench.sh
#                (--hash keccak --log-permutations <LP>), run fresh each time.
#                The upstream report's "time" is prover time → shown in the
#                "prove" column; its patched bench crate also emits verifier time.
# Each prover runs its own per-mode size list (flock too, at its 3·2^(h-1) sweet
# spots). Result reuse is handled uniformly by the bench-keccak-cache/ layer (see
# "Caching" below), not by any per-prover report cache.
#
# Sweeps HASH_LOG2S for binius64/plonky3:
#   - binius64 proves EXACTLY 2^N independent permutations (N_PERMUTATIONS=2^N).
#   - plonky3 needs a power-of-2 trace height (24 rows/keccak), so it cannot hit
#     2^N exactly; it uses LOG_TRACE_LENGTH=N+5, the smallest trace >= 2^N*24,
#     which yields ~1.33x the target (e.g. 2^12 → 5461 keccaks). The "keccaks"
#     column shows each prover's ACTUAL proven count.
#
# NOTE on sizes: the default sweep is "12 14 16 18 19", but each prover only runs
# sizes up to its own memory cap (see <PROVER>_MAX_LOG2 below), so the report has
# all four provers at 2^12/2^14, then flock + hashcaster at 2^16, and hashcaster
# alone at 2^18/2^19 — the largest points that fit a 24 GB box. Peak memory:
# plonky3 2^14 → LOG=19 ~15 GB (2^16 → LOG 21 needs >40 GB, capped out); flock
# (3-wide) ~14 GB @ 2^16's 98304 keccaks (2^18 capped out); hashcaster ~1.4 GB @
# 2^16, ~5.6 GB @ 2^18, ~11 GB @ 2^19; binius64 2^16 ≈ 8 GB is feasible but off by
# default (B64_MAX_LOG2=14, and the frontend build is impractically slow there
# anyway) — Raise the caps (and/or RAM) for larger runs.
#
# Usage: bench_keccak.sh [PROVER ...]   (no args = run all). PROVER is any subset of:
#   flock | flock-slim | flock-secure | binius64 | plonky3 | hashcaster
#   bhs   (= hashcaster)
#   both  (= binius64 plonky3)    all (default)
# (flock-slim = the same 3-wide Flock prover at PCS rate 1/4 — smaller proof,
#  slower prover; runs ONLY at 2^14 by default (FLOCK_SLIM_LOG2S) as a fast-vs-slim
#  contrast point, not a full sweep. Uses the keccak3_slim_proof bench.)
# (flock-secure = the same 3-wide Flock prover at the default rate but the audited
#  120-bit-security configs (m*_secure.toml) — larger proof, higher provable
#  security; runs ONLY at 2^14 by default (FLOCK_SECURE_LOG2S). Uses the
#  keccak3_secure_proof bench.)
# (plonky3-bhs and expander are disabled by default — plonky3-bhs is superseded
#  by `plonky3`; expander needs a system MPI toolchain (mpicc) and doesn't build
#  here. Both are commented out below; re-enable via their case + sweep entries.)
#   e.g.  ./bench_keccak.sh flock hashcaster        ./bench_keccak.sh bhs
# (The $ONLY env var still works too, and may be space-separated.)
#
# Caching: every result is saved under bench-keccak-cache/<prover>_2^<h>_t<threads>. A plain
# `./bench_keccak.sh` (no args, no $ONLY) reuses cached rows and only runs what's
# missing; naming a prover re-runs it fresh and refreshes its cache. USE_CACHE=1
# reuses the cache even with args; NO_CACHE=1 forces fresh runs (still caching).
#
# Single-threaded comparison: after the main (P-core) pass, every selected prover
# is re-run single-threaded (RAYON_NUM_THREADS=1) over its own *_ST_SIZES list and
# printed as a second table. ST_LOG2S is an on/off toggle (set ST_LOG2S="" to skip
# the whole pass); it's also auto-skipped when the main pass is already
# single-threaded.
#
# Sizes are per-prover, per-mode lists ({FLOCK,FLOCK_SLIM,B64,P3,HC}_{MT,ST}_SIZES,
# defaults below): the provers can cover different ranges and a prover's MT and ST
# lists can differ. Other knobs:
# RAYON_NUM_THREADS, <PROVER>_MAX_LOG2 caps, sub-scripts' own knobs (KECCAK_MODE,
# ...). See ../CLAUDE.md "External Benchmarks".

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOCK_ROOT="$(cd "$BASE/.." && pwd)"
BHS_DIR="$BASE/bench-hash-in-snark"

# Thread count for the bench-hash-in-snark runs — also part of their cached
# report-file names (report/t<THREADS>_keccak_lp<LP>). Defaults to the
# performance-core count (matches the other provers); RAYON_NUM_THREADS wins.
if [[ -n "${RAYON_NUM_THREADS:-}" ]]; then
	THREADS="$RAYON_NUM_THREADS"
else
	if [[ "$(uname -s)" == "Darwin" ]]; then PCORES="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)"; fi
	: "${PCORES:=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
	THREADS="$PCORES"
fi
PRIMARY_THREADS="$THREADS"   # the main (multi-thread) pass; the single-thread pass overrides THREADS=1

# Which provers to run. Specify any subset as CLI args (or via $ONLY, which may
# be space-separated); with none given, run them all. Names + group aliases:
#   flock | flock-slim | flock-secure | binius64 | plonky3 | hashcaster   (plonky3-bhs, expander disabled)
#   bhs (= hashcaster)   both (= binius64 plonky3)   all
# e.g.  ./bench_keccak.sh flock hashcaster      ONLY="flock hashcaster" ./bench_keccak.sh
# (flock is the 3-wide keccak encoder — see run_flock.)

# --cooldown N (seconds to sleep between sweeps; default 0 = off; also via the
# COOLDOWN env). Stripped from the positional args here, before prover selection
# and the cache-read decision below. See the "Cooldowns" helpers further down.
COOLDOWN="${COOLDOWN:-0}"
_pos=()
while [[ $# -gt 0 ]]; do
	case "$1" in
		--cooldown) shift; COOLDOWN="${1:-0}"; [[ $# -gt 0 ]] && shift ;;
		--cooldown=*) COOLDOWN="${1#*=}"; shift ;;
		*) _pos+=("$1"); shift ;;
	esac
done
set -- ${_pos[@]+"${_pos[@]}"}
[[ "$COOLDOWN" =~ ^[0-9]+$ ]] || { echo "--cooldown expects a non-negative integer (seconds), got '$COOLDOWN'" >&2; exit 1; }

do_flock=false; do_flock_slim=false; do_flock_secure=false; do_b64=false; do_p3=false; do_exp=false; do_hc=false; do_p3bhs=false
if [[ $# -gt 0 ]]; then
	REQUESTED=("$@")
elif [[ -n "${ONLY:-}" ]]; then
	# shellcheck disable=SC2206
	REQUESTED=($ONLY)
else
	REQUESTED=(all)
fi
for tok in "${REQUESTED[@]}"; do
	case "$tok" in
		flock)        do_flock=true ;;
		flock-slim)   do_flock_slim=true ;;
		flock-secure) do_flock_secure=true ;;
		binius64)    do_b64=true ;;
		plonky3)     do_p3=true ;;
		hashcaster)  do_hc=true ;;
		# expander disabled: PolyhedraZK GKR needs a system MPI toolchain (mpicc)
		# and doesn't build here. Uncomment this line, the `do_exp=true` in
		# `bhs`/`all` below, and the run_bhs call in the sweep loop to re-enable.
		# expander)    do_exp=true ;;
		# plonky3-bhs disabled: superseded by the `plonky3` row (real Plonky3
		# example — newer, high-arity FRI: smaller proof + faster verify at the
		# same security/throughput). Uncomment this line, the `do_p3bhs=true` in
		# `bhs`/`all` below, and the run_bhs call in the sweep loop to re-enable.
		# plonky3-bhs) do_p3bhs=true ;;
		bhs)         do_hc=true ;;
		both)        do_b64=true; do_p3=true ;;
		all)         do_flock=true; do_flock_slim=true; do_flock_secure=true; do_b64=true; do_p3=true; do_hc=true ;;
		*) echo "unknown prover '$tok' (flock|flock-slim|flock-secure|binius64|plonky3|hashcaster|bhs|both|all)" >&2; exit 1 ;;
	esac
done

# Per-prover, per-mode size lists (log2 keccak target 2^h; flock's 3-wide encoder
# proves 3·2^(h-1) keccaks at target 2^h). Each prover runs exactly its own list
# in each pass (see sweep_provers) — there is no shared sweep, so the provers can
# cover different ranges (flock/hashcaster at the large sizes, the competitors at
# the small ones), and a prover's MT and ST lists can differ. Sizes above a
# prover's memory cap (below) are still skipped as a safety net.
FLOCK_MT_SIZES="${FLOCK_MT_SIZES:-12 14 16}"
FLOCK_SLIM_MT_SIZES="${FLOCK_SLIM_MT_SIZES:-14}"
FLOCK_SECURE_MT_SIZES="${FLOCK_SECURE_MT_SIZES:-14}"
B64_MT_SIZES="${B64_MT_SIZES:-12 14}"
P3_MT_SIZES="${P3_MT_SIZES:-10 12 14}"
HC_MT_SIZES="${HC_MT_SIZES:-14 16 18}"

FLOCK_ST_SIZES="${FLOCK_ST_SIZES:-10 12 14}"
FLOCK_SLIM_ST_SIZES="${FLOCK_SLIM_ST_SIZES:-}"   # flock-slim: no single-threaded row
FLOCK_SECURE_ST_SIZES="${FLOCK_SECURE_ST_SIZES:-}"   # flock-secure: no single-threaded row
B64_ST_SIZES="${B64_ST_SIZES:-10 12 14}"
P3_ST_SIZES="${P3_ST_SIZES:-10 12 14}"
HC_ST_SIZES="${HC_ST_SIZES:-10 12 14}"

# Plonky3-ONLY extra multi-threaded sizes (within its cap, so no force needed).
# The master sweep only hits plonky3 at 2^12/2^14; these fill in the smaller MT
# points so the multi-threaded table has a fuller plonky3 curve (2^10..2^14)
# without dragging the other provers down to those sizes. Run in the main (MT)
# pass. Set P3_MT_EXTRA="" to skip.
P3_MT_EXTRA="${P3_MT_EXTRA-10 11 13}"

# Flock-ONLY extra multi-threaded sizes, above its default cap (FLOCK_MAX_LOG2).
# Unlike binius64, flock's 3-wide prover genuinely fits beyond the cap — 2^17
# proves 3·2^16 = 196608 keccaks at ~13 GB (the prover scales ~2x per +1, from
# ~6.7 GB at 2^16). Run in the main (MT) pass with the cap lifted. The cap stays
# 16 so the *default sweep* (which has no 2^17) doesn't change; this is the opt-in
# way to push flock one size higher. Set FLOCK_MT_EXTRA="" to skip. 2^18
# (3·2^17 ≈ 26 GB) would not fit a 24 GB box.
FLOCK_MT_EXTRA="${FLOCK_MT_EXTRA-17}"

# Binius64-ONLY extra multi-threaded sizes, above its default cap (B64_MAX_LOG2).
# Run with the cap forced off (run_b64_size "$h" force) in the main (MT) pass.
# Empty by default: binius64's frontend build doesn't fit a 24 GB box above 2^14
# (2^14 build peaks ~34 GB, 2^15 ~65 GB thrashes/hangs — see B64_MAX_LOG2 notes
# below), so there is no extra size to add unless you're on a bigger box. Set
# e.g. B64_MT_EXTRA="15" to opt in there.
B64_MT_EXTRA="${B64_MT_EXTRA-}"

# ST_LOG2S — on/off TOGGLE for the single-threaded pass (non-empty = run it,
# "" = skip). It no longer sets the single-threaded sizes: those now come from
# each prover's own *_ST_SIZES list (sweep_provers picks the list by mode, not
# from this value). The default string is just a non-empty sentinel. To change
# which sizes run single-threaded, edit FLOCK_ST_SIZES / B64_ST_SIZES / … above.
# The ST pass is also auto-skipped when the main pass is already single-threaded.
# (No colon in the expansion: an explicit ST_LOG2S="" stays empty = skip.)
ST_LOG2S="${ST_LOG2S-on}"

# Plonky3-ONLY extra single-threaded sizes, below the common ST floor. Plonky3's
# per-core throughput keeps rising as the batch shrinks (it peaks around 2^8-2^9
# and only turns over at 2^7), whereas every other prover declines at tiny
# batches — so we sweep Plonky3 down to 2^7 but spare the others uninformative
# tail rows. Run in the single-threaded pass only. Set P3_ST_EXTRA="" to skip.
P3_ST_EXTRA="${P3_ST_EXTRA-7 8 9}"

# Per-prover max log2 size (peak-memory ceilings on a 24 GB M4 Max; raise on a
# bigger box). A prover skips any sweep size above its cap, with a stderr notice.
#   plonky3   ~14: 2^16 → LOG_TRACE 21 needs >40 GB.
#   binius64  ~14: limited by its frontend's TRANSIENT build footprint, not the
#                  prover. CircuitBuilder::build() peaks at ~18 GB @ 2^13, ~34 GB
#                  @ 2^14 (vs the prover's ~2.4 GB), then is freed; it exceeds the
#                  24 GB RAM from 2^14 on (compressor → ~70 s build), and 2^15
#                  (~65 GB) thrashes/hangs in build before proving. See B64_MT_EXTRA.
#   flock      16: 3-wide encoder, ~14 GB at 2^16's 3·2^15 = 98304 keccaks;
#                  2^18 would not fit. (FLOCK_MAX_LOG2)
#   hashcaster 19: ~5.6 GB @ 2^18, ~11 GB @ 2^19 (~2x per +1); ~22 GB @ 2^20.
P3_MAX_LOG2="${P3_MAX_LOG2:-14}"
B64_MAX_LOG2="${B64_MAX_LOG2:-14}"
HC_MAX_LOG2="${HC_MAX_LOG2:-19}"
# FLOCK_MAX_LOG2 (default 16) is read inside run_flock.

# Each ROWS entry is TAB-separated: sortkey, name, target, keccaks, throughput,
# prove_s, verify, size, peakmem. sortkey = printf "%d%03d" <prover-order>
# <log2-size> so a plain `sort` groups the report BY PROVER (flock < binius64 <
# plonky3 < hashcaster), then by target keccaks (size) ascending
# within each prover, regardless of run order.
ROWS=()

# --- Result cache ------------------------------------------------------------
# Every obtained (prover, size) result is saved to a one-line file
# bench-keccak-cache/<prover>_2^<h>_t<threads> (the exact ROWS entry). A plain default run
# (no positional args and no $ONLY) reuses cached rows instead of re-running, so
# the full table assembles instantly once each point has been measured once.
# Naming a prover explicitly (e.g. `./bench_keccak.sh flock`) re-runs it fresh and
# refreshes its cache. Overrides: USE_CACHE=1 reuses the cache even with args;
# NO_CACHE=1 forces fresh runs (still rewriting the cache). Delete bench-keccak-cache/ (or the
# relevant file) to force re-measurement after changing a prover's config.
CACHE_DIR="$BASE/bench-keccak-cache"
mkdir -p "$CACHE_DIR"
if [[ -n "${NO_CACHE:-}" ]]; then
	cache_read=false
elif [[ -n "${USE_CACHE:-}" ]]; then
	cache_read=true
elif [[ $# -eq 0 && -z "${ONLY:-}" ]]; then
	cache_read=true
else
	cache_read=false
fi

# ---- Cooldowns (opt-in via --cooldown N / COOLDOWN=N) ------------------------
# Sleep N seconds between the multi-threaded and single-threaded sweeps to let
# the machine cool, so thermal throttling doesn't bias the heat-sensitive
# single-threaded pass. The cooldown only fires when the MT pass actually
# measured something fresh (a cache hit generates no heat), so a fully cached
# run never waits. add_row sets the flag; the ST cooldown reads and clears it.
COOLDOWN_DID_FRESH=0
mark_fresh() { COOLDOWN_DID_FRESH=1; }
cooldown() {
	[[ "${COOLDOWN:-0}" -gt 0 && "$COOLDOWN_DID_FRESH" == 1 ]] || return 0
	echo "  cooldown: sleeping ${COOLDOWN}s before ${1:-the next sweep} to let the machine cool..." >&2
	sleep "$COOLDOWN"
	COOLDOWN_DID_FRESH=0
}

cache_file() { echo "$CACHE_DIR/${1}_2^${2}_t${THREADS}"; }

# cache_lookup PROVER H — when cache-read is on and a cached row exists, append
# it to ROWS, print a note, and return 0 (caller skips running); else return 1.
cache_lookup() {
	local f; f="$(cache_file "$1" "$2")"
	[[ "$cache_read" == true && -s "$f" ]] || return 1
	echo "  using cached result: $1 2^$2 ($(( 1 << $2 ))), ${THREADS}t"
	ROWS+=("$(cat "$f")")
}

# add_row PROVER H ROW — record ROW in ROWS and cache it. FAILED rows aren't
# cached, so a transient error (OOM, missing dep) isn't pinned for later runs.
add_row() {
	ROWS+=("$3")
	mark_fresh   # a fresh measurement ran (cache hits never reach add_row)
	case "$3" in
		*$'\t'FAILED$'\t'*) ;;
		*) printf '%s\n' "$3" > "$(cache_file "$1" "$2")" ;;
	esac
}

# to_ms "VALUE UNIT" — normalize a "12.3 ms" / "0.012 s" / "45 µs" pair to ms.
to_ms() {
	awk -v s="${1:-}" 'BEGIN {
		n = split(s, a, " "); if (n < 2) { print "n/a"; exit }
		v = a[1]; u = a[2]
		mult = (u=="s") ? 1000 : (u=="ms") ? 1 : (u=="µs"||u=="us") ? 0.001 : (u=="ns") ? 1e-6 : 0
		if (mult == 0) { print "n/a"; exit }
		printf "%.2f ms", v * mult
	}'
}

# human_size BYTES — pretty-print a byte count (empty → "n/a").
human_size() {
	awk -v b="${1:-}" 'BEGIN {
		if (b == "") { print "n/a"; exit }
		if (b+0 < 1024)       printf "%d B", b
		else if (b+0 < 1048576) printf "%.2f KiB", b/1024
		else                  printf "%.2f MiB", b/1048576
	}'
}

# proof_bytes LOG — extract the proof size in bytes from a sub-script log.
# Handles "… N bytes (X KiB)" (binius) and "Proof size: N bytes" (plonky3).
proof_bytes() {
	awk 'tolower($0) ~ /proof size/ {
		line = $0
		if (match(line, /[0-9]+ bytes \(/)) { s = substr(line, RSTART, RLENGTH); sub(/ bytes \(/, "", s); v = s }
		else { tmp = line; while (match(tmp, /[0-9]+ bytes/)) { s = substr(tmp, RSTART, RLENGTH); sub(/ bytes/, "", s); v = s; tmp = substr(tmp, RSTART + RLENGTH) } }
	} END { print v }' "$1"
}

# run_one KEY NAME SCRIPT TARGET — run a sub-script (env exported by caller),
# stream its output live, and record (keccaks, throughput, prove_s, size).
# Resilient: a failing sub-script (e.g. OOM) is recorded, not fatal to the sweep.
run_one() {
	local key="$1" name="$2" script="$3" target="$4" log status line thr kc prove vfy sz mem h
	h=$((10#${key#?}))   # key = printf '%d%03d' <order> <h>; strip the leading order digit
	echo
	echo "############### $name — $target ###############"
	log="$(mktemp)"
	set +e
	bash "$script" 2>&1 | tee "$log"
	status="${PIPESTATUS[0]}"
	set -e
	if [[ "$status" -ne 0 ]]; then
		add_row "$name" "$h" "$key"$'\t'"$name"$'\t'"$target"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
		rm -f "$log"; return
	fi
	line="$(grep -E 'throughput:' "$log" | tail -1 | sed 's/\x1b\[[0-9;]*m//g')"
	read -r thr kc prove < <(awk '{
		for (i = 1; i <= NF; i++) {
			if ($i == "throughput:")  t = $(i+1)
			if ($i == "permutations") k = $(i-1)
			if ($i == "/")            p = $(i+1)
		}
		print t, k, p
	}' <<< "$line")
	# verify time (binius64 prints "verify: X s"); peak memory ("peak memory: X MB").
	vfy="$(to_ms "$(awk '$0 ~ /verify:/ { for (i=1;i<=NF;i++) if ($i=="verify:") { print $(i+1)" "$(i+2); exit } }' "$log")")"
	mem="$(awk '/peak memory:/ { for (i=1;i<=NF;i++) if ($i=="memory:") { print $(i+1)" "$(i+2); exit } }' "$log")"
	sz="$(human_size "$(proof_bytes "$log")")"
	rm -f "$log"
	add_row "$name" "$h" "$key"$'\t'"$name"$'\t'"$target"$'\t'"${kc:-?}"$'\t'"${thr:-?}"$'\t'"${prove:-?}"$'\t'"$vfy"$'\t'"$sz"$'\t'"${mem:-n/a}"
}

# flock_secs N LOG — prove_fast time (seconds) of the K=N section, or empty.
flock_secs() {
	awk -v want="$1" '
		/=== K =/ { for (i = 1; i <= NF; i++) if ($i == "Keccaks") { curN = $(i-1); break } }
		curN == want && /best prove_fast:/ {
			for (i = 1; i <= NF; i++) if ($i == "prove_fast:") { val = $(i+1); u = $(i+2); break }
			mult = (u=="s") ? 1 : (u=="ms") ? 1e-3 : (u=="µs"||u=="us") ? 1e-6 : (u=="ns") ? 1e-9 : 0
			if (mult > 0) { printf "%.6f", val * mult; exit }
		}' "$2"
}

# flock_field N LOG TOKEN — within the K=N section, print "<next> <next2>" after
# the field named TOKEN (e.g. "verify:" → "3.75 ms", "memory:" → "445.76 MB").
flock_field() {
	awk -v want="$1" -v tok="$3" '
		/=== K =/ { for (i = 1; i <= NF; i++) if ($i == "Keccaks") { curN = $(i-1); break } }
		curN == want { for (i = 1; i <= NF; i++) if ($i == tok) { print $(i+1)" "$(i+2); exit } }' "$2"
}

# flock_proof_bytes N LOG — proof size in bytes from the K=N "proof size:" line.
flock_proof_bytes() {
	awk -v want="$1" '
		/=== K =/ { for (i = 1; i <= NF; i++) if ($i == "Keccaks") { curN = $(i-1); break } }
		curN == want && /proof size:/ { for (i = 1; i <= NF; i++) if ($i == "size:") { print $(i+1); exit } }' "$2"
}

# run_flock — Flock's keccak prover (the 3-wide keccak3 encoder), swept over
# $HASH_LOG2S via the keccak3_proof bench's KECCAK3_KS mode. "best prove_fast" =
# full fast prover (incl. witness gen), parsed by the flock_* helpers. The 3-wide
# encoder packs three independent permutations per commitment block (~97% block
# utilization vs a naive ~65%), but only wins at counts of the form 3·2^j; at
# exact powers of two it merely ties (or slightly trails) the naive encoder. So
# at each 2^h comparison point it proves N = 3·2^(h-1) keccaks: that lands the
# encoder on exactly 2^(h-1) blocks, committed size m = K_LOG + (h-1) = 16 + h —
# i.e. it proves 1.5x the keccaks (3·2^(h-1) = 1.5·2^h) at the SAME commitment
# cost as a naive 2^h-block encoder. The "target" column shows the 2^h point; the
# "keccaks" column shows the actual 3·2^(h-1) proven (like hashcaster, which also
# lands on 1.5x its 2^h target). Sizes above FLOCK_MAX_LOG2 (default 16) are
# skipped (peak memory; ~14 GB at 2^16's 3·2^15 = 98304).
run_flock() {
	local log status spec h n key v thr prove vfy mem sz specs ks
	local max="${FLOCK_MAX_LOG2:-16}"
	specs=(); ks=""
	for h in $HASH_LOG2S; do
		(( h >= 1 )) || continue   # need h-1 >= 0 for 3·2^(h-1)
		if (( h > max )); then
			echo "skipping flock 2^$h: peak memory exceeds FLOCK_MAX_LOG2=$max budget (~24 GB); set FLOCK_MAX_LOG2 higher on a bigger box." >&2
			continue
		fi
		cache_lookup flock "$h" && continue
		n=$(( 3 * (1 << (h - 1)) ))
		specs+=("$h:$n"); ks="${ks:+$ks }$n"
	done
	if [[ -z "$ks" ]]; then
		return   # everything cached or over budget; cached rows already appended
	fi
	cooldown "flock 2^${specs[0]%%:*}"
	echo
	echo "############### flock — keccak3_proof (K=[$ks]) ###############"
	log="$(mktemp)"
	set +e
	( cd "$FLOCK_ROOT" && KECCAK3_KS="$ks" cargo bench --bench keccak3_proof ) 2>&1 | tee "$log"
	status="${PIPESTATUS[0]}"
	set -e
	if [[ "$status" -ne 0 ]]; then
		for spec in "${specs[@]}"; do
			h="${spec%%:*}"; n="${spec##*:}"
			add_row flock "$h" "$(printf '%d%03d' 0 "$h")"$'\t'"flock"$'\t'"2^$h ($(( 1 << h )))"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
		done
		rm -f "$log"; return
	fi
	for spec in "${specs[@]}"; do
		h="${spec%%:*}"; n="${spec##*:}"
		key="$(printf '%d%03d' 0 "$h")"   # flock = prover-order 0
		vfy="$(to_ms "$(flock_field "$n" "$log" "verify:")")"
		mem="$(flock_field "$n" "$log" "memory:")"; mem="${mem:-n/a}"
		sz="$(human_size "$(flock_proof_bytes "$n" "$log")")"
		v="$(flock_secs "$n" "$log")"
		if [[ -z "$v" ]]; then
			ROWS+=("$key"$'\t'"flock"$'\t'"2^$h ($(( 1 << h )))"$'\t'"$n"$'\t'"?"$'\t'"?"$'\t'"$vfy"$'\t'"$sz"$'\t'"$mem")  # parse failed; don't cache
		else
			thr="$(awk -v v="$v" -v n="$n" 'BEGIN { printf "%.0f", n / v }')"
			prove="$(awk -v v="$v" 'BEGIN { printf "%.3f", v }')"
			add_row flock "$h" "$key"$'\t'"flock"$'\t'"2^$h ($(( 1 << h )))"$'\t'"$n"$'\t'"$thr"$'\t'"$prove"$'\t'"$vfy"$'\t'"$sz"$'\t'"$mem"
		fi
	done
	rm -f "$log"
}

# run_flock_slim — the SLIM Flock keccak prover: same 3-wide encoder but PCS rate
# 1/4 (keccak3_slim_proof, log_inv_rate=2) → smaller proofs, slower prover. Runs
# ONLY at the sizes in FLOCK_SLIM_LOG2S (default "14"): it's a fixed-point
# proof-size/throughput contrast against the default (fast, rate 1/2) flock, not
# a full sweep. Same 3·2^(h-1) count and output format as run_flock (so the
# flock_* parsers apply); sorts just after flock (prover-order 1).
run_flock_slim() {
	local log status spec h n key v thr prove vfy mem sz specs ks
	local max="${FLOCK_MAX_LOG2:-16}"
	local only="${FLOCK_SLIM_LOG2S:-14}"
	specs=(); ks=""
	for h in $HASH_LOG2S; do
		case " $only " in *" $h "*) ;; *) continue ;; esac   # slim sizes only (default 2^14)
		(( h >= 1 )) || continue
		if (( h > max )); then
			echo "skipping flock-slim 2^$h: above FLOCK_MAX_LOG2=$max budget." >&2
			continue
		fi
		cache_lookup flock-slim "$h" && continue
		n=$(( 3 * (1 << (h - 1)) ))
		specs+=("$h:$n"); ks="${ks:+$ks }$n"
	done
	if [[ -z "$ks" ]]; then
		return
	fi
	cooldown "flock-slim 2^${specs[0]%%:*}"
	echo
	echo "############### flock-slim — keccak3_slim_proof (K=[$ks]) ###############"
	log="$(mktemp)"
	set +e
	( cd "$FLOCK_ROOT" && KECCAK3_KS="$ks" cargo bench --bench keccak3_slim_proof ) 2>&1 | tee "$log"
	status="${PIPESTATUS[0]}"
	set -e
	if [[ "$status" -ne 0 ]]; then
		for spec in "${specs[@]}"; do
			h="${spec%%:*}"; n="${spec##*:}"
			add_row flock-slim "$h" "$(printf '%d%03d' 1 "$h")"$'\t'"flock-slim"$'\t'"2^$h ($(( 1 << h )))"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
		done
		rm -f "$log"; return
	fi
	for spec in "${specs[@]}"; do
		h="${spec%%:*}"; n="${spec##*:}"
		key="$(printf '%d%03d' 1 "$h")"   # flock-slim = prover-order 1 (just after flock)
		vfy="$(to_ms "$(flock_field "$n" "$log" "verify:")")"
		mem="$(flock_field "$n" "$log" "memory:")"; mem="${mem:-n/a}"
		sz="$(human_size "$(flock_proof_bytes "$n" "$log")")"
		v="$(flock_secs "$n" "$log")"
		if [[ -z "$v" ]]; then
			ROWS+=("$key"$'\t'"flock-slim"$'\t'"2^$h ($(( 1 << h )))"$'\t'"$n"$'\t'"?"$'\t'"?"$'\t'"$vfy"$'\t'"$sz"$'\t'"$mem")  # parse failed; don't cache
		else
			thr="$(awk -v v="$v" -v n="$n" 'BEGIN { printf "%.0f", n / v }')"
			prove="$(awk -v v="$v" 'BEGIN { printf "%.3f", v }')"
			add_row flock-slim "$h" "$key"$'\t'"flock-slim"$'\t'"2^$h ($(( 1 << h )))"$'\t'"$n"$'\t'"$thr"$'\t'"$prove"$'\t'"$vfy"$'\t'"$sz"$'\t'"$mem"
		fi
	done
	rm -f "$log"
}

# run_flock_secure — the SECURE Flock keccak prover: same 3-wide encoder at the
# default PCS rate (1/2), but the audited 120-bit-security configs
# (keccak3_secure_proof, Ligerito Secure profile, m*_secure.toml) → larger
# proof, ~the same prover work, a higher provable-security target. Runs ONLY at
# the sizes in FLOCK_SECURE_LOG2S (default "14"): a fixed-point 120-bit contrast
# against the default (fast, 100-bit) flock, not a full sweep. Same 3·2^(h-1)
# count and output format as run_flock (so the flock_* parsers apply); sorts
# just after flock-slim (prover-order 2).
run_flock_secure() {
	local log status spec h n key v thr prove vfy mem sz specs ks
	local max="${FLOCK_MAX_LOG2:-16}"
	local only="${FLOCK_SECURE_LOG2S:-14}"
	specs=(); ks=""
	for h in $HASH_LOG2S; do
		case " $only " in *" $h "*) ;; *) continue ;; esac   # secure sizes only (default 2^14)
		(( h >= 1 )) || continue
		if (( h > max )); then
			echo "skipping flock-secure 2^$h: above FLOCK_MAX_LOG2=$max budget." >&2
			continue
		fi
		cache_lookup flock-secure "$h" && continue
		n=$(( 3 * (1 << (h - 1)) ))
		specs+=("$h:$n"); ks="${ks:+$ks }$n"
	done
	if [[ -z "$ks" ]]; then
		return
	fi
	cooldown "flock-secure 2^${specs[0]%%:*}"
	echo
	echo "############### flock-secure — keccak3_secure_proof (K=[$ks]) ###############"
	log="$(mktemp)"
	set +e
	( cd "$FLOCK_ROOT" && KECCAK3_KS="$ks" cargo bench --bench keccak3_secure_proof ) 2>&1 | tee "$log"
	status="${PIPESTATUS[0]}"
	set -e
	if [[ "$status" -ne 0 ]]; then
		for spec in "${specs[@]}"; do
			h="${spec%%:*}"; n="${spec##*:}"
			add_row flock-secure "$h" "$(printf '%d%03d' 2 "$h")"$'\t'"flock-secure"$'\t'"2^$h ($(( 1 << h )))"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
		done
		rm -f "$log"; return
	fi
	for spec in "${specs[@]}"; do
		h="${spec%%:*}"; n="${spec##*:}"
		key="$(printf '%d%03d' 2 "$h")"   # flock-secure = prover-order 2 (just after flock-slim)
		vfy="$(to_ms "$(flock_field "$n" "$log" "verify:")")"
		mem="$(flock_field "$n" "$log" "memory:")"; mem="${mem:-n/a}"
		sz="$(human_size "$(flock_proof_bytes "$n" "$log")")"
		v="$(flock_secs "$n" "$log")"
		if [[ -z "$v" ]]; then
			ROWS+=("$key"$'\t'"flock-secure"$'\t'"2^$h ($(( 1 << h )))"$'\t'"$n"$'\t'"?"$'\t'"?"$'\t'"$vfy"$'\t'"$sz"$'\t'"$mem")  # parse failed; don't cache
		else
			thr="$(awk -v v="$v" -v n="$n" 'BEGIN { printf "%.0f", n / v }')"
			prove="$(awk -v v="$v" 'BEGIN { printf "%.3f", v }')"
			add_row flock-secure "$h" "$key"$'\t'"flock-secure"$'\t'"2^$h ($(( 1 << h )))"$'\t'"$n"$'\t'"$thr"$'\t'"$prove"$'\t'"$vfy"$'\t'"$sz"$'\t'"$mem"
		fi
	done
	rm -f "$log"
}

# --- bench-hash-in-snark helpers ---------------------------------------------
# bhs_field REPORT TOKEN — print "<next> <next2>" after the field named TOKEN.
bhs_field() { awk -v tok="$2" '{ for (i=1;i<=NF;i++) if ($i==tok) { print $(i+1)" "$(i+2); exit } }' "$1"; }
# bhs_per_s "5.30 K/s" → keccaks/sec (number); bhs_s "1.03 s" → seconds.
bhs_per_s() { awk -v s="${1:-}" 'BEGIN { if (split(s,a," ")<2) exit; m=(a[2]=="M/s")?1e6:(a[2]=="K/s")?1e3:(a[2]=="/s")?1:0; if (m>0) printf "%.3f", a[1]*m }'; }
bhs_s()     { awk -v s="${1:-}" 'BEGIN { if (split(s,a," ")<2) exit; m=(a[2]=="s")?1:(a[2]=="ms")?1e-3:(a[2]=="µs"||a[2]=="us")?1e-6:(a[2]=="ns")?1e-9:0; if (m>0) printf "%.6f", a[1]*m }'; }
# bhs_size_bytes "4.19 MB" → bytes (upstream human_size uses KB=2^10, MB=2^20).
bhs_size_bytes() { awk -v s="${1:-}" 'BEGIN { if (split(s,a," ")<2) exit; m=(a[2]=="GB")?1073741824:(a[2]=="MB")?1048576:(a[2]=="KB")?1024:(a[2]=="B")?1:0; if (m>0) printf "%d", a[1]*m }'; }

# run_bhs KEY LABEL PACKAGE LP — run a bench-hash-in-snark keccak benchmark fresh
# and record its result. The upstream bench.sh writes its numbers to
# report/t<THREADS>_keccak_lp<LP>, which we parse but never reuse as a cache (the
# outer bench-keccak-cache/ layer handles result reuse). bench-hash-in-snark's patched bench
# crate emits a "verify time:" line; absent → verify = n/a.
run_bhs() {
	local key="$1" label="$2" pkg="$3" lp="$4"
	local n report status thr_s t_s thr kc prove vfy sz mem d mf
	n=$(( 1 << lp ))
	report="$BHS_DIR/$pkg/report/t${THREADS}_keccak_lp${lp}"
	echo
	echo "############### $label — 2^$lp ($n) ###############"
	if [[ ! -f "$BHS_DIR/bench.sh" ]]; then
		echo "bench-hash-in-snark not present — run benchmarks/bench-hash-in-snark/setup.sh first." >&2
		add_row "$label" "$lp" "$key"$'\t'"$label"$'\t'"2^$lp ($n)"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
		return
	fi
	# expander (PolyhedraZK GKR) needs a system MPI toolchain (mpi-sys/libffi).
	# Skip with a clear note instead of attempting a doomed build every run.
	if [[ "$pkg" == "expander" ]] && ! command -v mpicc >/dev/null 2>&1; then
		echo "skipping expander: needs an MPI toolchain (mpicc) — e.g. 'brew install open-mpi'." >&2
		add_row "$label" "$lp" "$key"$'\t'"$label"$'\t'"2^$lp ($n)"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
		return
	fi
	# Detach from any ancestor cargo workspace (flock-rust/Cargo.toml).
	# expander/hashcaster pin old nightlies whose cargo can't parse Flock's
	# edition-2024 manifest during the ancestor scan; an empty [workspace] table
	# makes a crate its own workspace root so cargo never reads the ancestor.
	# Detach the package AND the shared `bench` harness crate it depends on
	# (cargo would otherwise walk up from ../bench too).
	for d in "$pkg" bench; do
		mf="$BHS_DIR/$d/Cargo.toml"
		if [[ -f "$mf" ]] && ! grep -q '^\[workspace\]' "$mf"; then
			printf '\n[workspace]\n' >> "$mf"
		fi
	done
	# Always run fresh: drop any prior report so a failed run can't be parsed as a
	# stale result, then regenerate it.
	rm -f "$report"
	echo "running bench-hash-in-snark: $pkg keccak lp=$lp (threads=$THREADS) ..."
	set +e
	( cd "$BHS_DIR" && RAYON_NUM_THREADS="$THREADS" PCS_LOG_INV_RATE="${PCS_LOG_INV_RATE:-1}" bash bench.sh "$pkg" keccak "$lp" )
	status=$?
	set -e
	if [[ "$status" -ne 0 || ! -f "$report" ]]; then
		add_row "$label" "$lp" "$key"$'\t'"$label"$'\t'"2^$lp ($n)"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
		return
	fi
	thr_s="$(bhs_per_s "$(bhs_field "$report" "throughput:")")"
	t_s="$(bhs_s "$(bhs_field "$report" "time:")")"
	sz="$(human_size "$(bhs_size_bytes "$(bhs_field "$report" "size:")")")"
	mem="$(bhs_field "$report" "mem:")"; mem="${mem:-n/a}"
	# verifier time (the patched bench crate emits a "verify time:" line; absent → n/a).
	vfy="$(to_ms "$(awk '/verify time:/ { for (i=1;i<=NF;i++) if ($i=="time:") { print $(i+1)" "$(i+2); exit } }' "$report")")"
	if [[ -z "$thr_s" || -z "$t_s" ]]; then
		# Report exists but has no throughput line (run failed — e.g. missing
		# system dep). Mark FAILED rather than emitting a misleading partial row.
		add_row "$label" "$lp" "$key"$'\t'"$label"$'\t'"2^$lp ($n)"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
		return
	fi
	thr="$(awk -v t="$thr_s" 'BEGIN { printf "%.0f", t }')"
	# True permutation count from the patched harness ("permutations: N"); the
	# actual proven count can differ from 2^lp (hashcaster rounds to 3*2^k = 1.5x).
	# Fall back to throughput*time for older reports without the line.
	kc="$(awk '/^permutations:/ { print $2; exit }' "$report")"
	[[ -n "$kc" ]] || kc="$(awk -v t="$thr_s" -v s="$t_s" 'BEGIN { printf "%.0f", t*s }')"
	prove="$(awk -v s="$t_s" 'BEGIN { printf "%.3f", s }')"
	add_row "$label" "$lp" "$key"$'\t'"$label"$'\t'"2^$lp ($n)"$'\t'"$kc"$'\t'"$thr"$'\t'"$prove"$'\t'"${vfy:-n/a}"$'\t'"$sz"$'\t'"$mem"
}

# run_p3_size H — run Plonky3 at log2 batch size H (honors P3_MAX_LOG2 + cache).
# Plonky3 needs a power-of-2 trace; LOG_TRACE_LENGTH = H+5 is the smallest trace
# >= 2^H * 24 rows. Factored out so the single-threaded pass can also sweep
# Plonky3 over its extra small sizes ($P3_ST_EXTRA) that other provers skip.
run_p3_size() {
	local h="$1" n
	n=$(( 1 << h ))
	if (( h > P3_MAX_LOG2 )); then
		echo "skipping plonky3 2^$h: above P3_MAX_LOG2=$P3_MAX_LOG2 budget (2^16 → LOG 21 needs >40 GB)." >&2
	elif cache_lookup plonky3 "$h"; then :
	else
		export LOG_TRACE_LENGTH="$(( h + 5 ))"
		run_one "$(printf '%d%03d' 4 "$h")" plonky3 "$BASE/plonky3/setup.sh" "2^$h ($n)"
	fi
}

# run_b64_size H [force] — run Binius64 at log2 batch size H (honors the result
# cache). Respects B64_MAX_LOG2 unless "force" is passed (used for the explicit
# binius64-only extra MT sizes in $B64_MT_EXTRA, which sit above the default cap).
run_b64_size() {
	local h="$1" n
	n=$(( 1 << h ))
	if [[ "${2:-}" != "force" ]] && (( h > B64_MAX_LOG2 )); then
		echo "skipping binius64 2^$h: above B64_MAX_LOG2=$B64_MAX_LOG2 budget." >&2
		return
	fi
	cache_lookup binius64 "$h" && return
	# Exact 2^N permutations (perm mode); HASH_MAX_BYTES set too so hash mode
	# (KECCAK_MODE=hash) also lands on 2^N permutations.
	export N_PERMUTATIONS="$n" HASH_MAX_BYTES="$(( n * 136 ))"
	run_one "$(printf '%d%03d' 3 "$h")" binius64 "$BASE/binius64/setup.sh" "2^$h ($n)"
}

# sweep_provers SIZES — run every selected prover over the given space-separated
# log2 sizes, at the current thread count (THREADS / RAYON_NUM_THREADS). Honors
# per-prover caps and the result cache. Used for both the main pass and the
# single-threaded pass.
sweep_provers() {  # mode: mt | st
	local mode="$1" fsz ssz xsz bsz psz hsz h n
	if [[ "$mode" == st ]]; then
		fsz="$FLOCK_ST_SIZES"; ssz="$FLOCK_SLIM_ST_SIZES"; xsz="$FLOCK_SECURE_ST_SIZES"; bsz="$B64_ST_SIZES"; psz="$P3_ST_SIZES"; hsz="$HC_ST_SIZES"
	else
		fsz="$FLOCK_MT_SIZES"; ssz="$FLOCK_SLIM_MT_SIZES"; xsz="$FLOCK_SECURE_MT_SIZES"; bsz="$B64_MT_SIZES"; psz="$P3_MT_SIZES"; hsz="$HC_MT_SIZES"
	fi
	# Each (prover, size) is its own run with a cooldown before it — the cooldowns
	# live inside the run_* functions, right before the real measurement, so
	# cap/cache skips don't trigger a pointless wait. flock/flock-slim/flock-secure
	# are driven per-size (one bench invocation each) so their sizes are separated too.
	if [[ "$do_flock"        == true ]]; then for h in $fsz; do HASH_LOG2S="$h" run_flock; done; fi
	if [[ "$do_flock_slim"   == true ]]; then for h in $ssz; do FLOCK_SLIM_LOG2S="$h" HASH_LOG2S="$h" run_flock_slim; done; fi
	if [[ "$do_flock_secure" == true ]]; then for h in $xsz; do FLOCK_SECURE_LOG2S="$h" HASH_LOG2S="$h" run_flock_secure; done; fi
	if [[ "$do_b64"        == true ]]; then for h in $bsz; do run_b64_size "$h"; done; fi
	if [[ "$do_p3"         == true ]]; then for h in $psz; do run_p3_size  "$h"; done; fi
	# hashcaster (bench-hash-in-snark). plonky3-bhs / expander are disabled — see
	# the ONLY case above to re-enable them.
	if [[ "$do_hc" == true ]]; then
		for h in $hsz; do
			if (( h > HC_MAX_LOG2 )); then
				echo "skipping hashcaster 2^$h: above HC_MAX_LOG2=$HC_MAX_LOG2 budget." >&2
			elif cache_lookup hashcaster "$h"; then :
			else
				run_bhs "$(printf '%d%03d' 5 "$h")" hashcaster  hashcaster "$h"
			fi
		done
	fi
}

# Main (multi-thread) pass over the full sweep. sweep_provers takes a MODE
# (mt|st), not sizes — each prover's sizes come from its own *_MT_SIZES /
# *_ST_SIZES list, not from HASH_LOG2S.
sweep_provers mt

# Binius64-only extra MT sizes (above its default cap; see B64_MT_EXTRA). Run in
# the main pass so they land in the multi-threaded table.
if [[ "$do_b64" == true && -n "$B64_MT_EXTRA" ]]; then
	for h_extra in $B64_MT_EXTRA; do run_b64_size "$h_extra" force; done
fi

# Plonky3-only extra MT sizes (within its cap; see P3_MT_EXTRA) — fills in the
# smaller multi-threaded points so plonky3 has a fuller MT curve.
if [[ "$do_p3" == true && -n "$P3_MT_EXTRA" ]]; then
	for h_extra in $P3_MT_EXTRA; do run_p3_size "$h_extra"; done
fi

# Flock-only extra MT sizes above its cap (see FLOCK_MT_EXTRA). run_flock honors
# FLOCK_MAX_LOG2, so lift it for this call only (these sizes are an explicit opt-in).
if [[ "$do_flock" == true && -n "$FLOCK_MT_EXTRA" ]]; then
	HASH_LOG2S="$FLOCK_MT_EXTRA" FLOCK_MAX_LOG2=99 run_flock
fi

# Single-threaded pass (RAYON_NUM_THREADS=1) over each prover's *_ST_SIZES list,
# gated by the ST_LOG2S on/off toggle. Its rows land in the tail of ROWS (from
# ST_START on) and are printed as a separate table. Skipped when the main pass is
# already single-threaded (would just duplicate it).
ST_START=${#ROWS[@]}
if [[ -n "$ST_LOG2S" && "$PRIMARY_THREADS" != "1" ]]; then
	echo
	echo "=== single-threaded pass (RAYON_NUM_THREADS=1): flock 2^[$FLOCK_ST_SIZES], binius64 2^[$B64_ST_SIZES], plonky3 2^[$P3_ST_SIZES], hashcaster 2^[$HC_ST_SIZES]$( [[ "$do_p3" == true && -n "$P3_ST_EXTRA" ]] && echo " + plonky3 2^[$P3_ST_EXTRA]" ) ==="
	THREADS=1
	export RAYON_NUM_THREADS=1
	cooldown "the keccak single-threaded pass"
	sweep_provers st
	# Plonky3-only extra small sizes (its per-core throughput peaks at tiny
	# batches; other provers decline there, so they don't run these).
	if [[ "$do_p3" == true && -n "$P3_ST_EXTRA" ]]; then
		for h_extra in $P3_ST_EXTRA; do run_p3_size "$h_extra"; done
	fi
fi

# sec_label NAME — targeted provable-security level in bits (a config property,
# not measured at run time). binius64's ~96 is FRI query-phase accounting.
sec_label() {
	case "$1" in
		flock)        echo "~100" ;;
		flock-slim)   echo "~100" ;;
		flock-secure) echo "~120" ;;
		binius64)    echo "~96"  ;;
		plonky3)     echo "~101" ;;   # 245 queries, 0 PoW (Plonky3 analyzer)
		plonky3-bhs) echo "~101" ;;   # 245 queries, 0 PoW
		hashcaster)  echo "100"  ;;   # security_bits = 100
		*)           echo "n/a"  ;;
	esac
}

# prover_order NAME — the prover's sort rank (table groups by prover). Must match
# the order digits baked into the run-time sortkeys; used by print_table to
# RE-derive the sort key from (name, target) so the ordering is correct even for
# rows loaded from the cache (whose embedded sortkey may predate an order change).
prover_order() {
	case "$1" in
		flock)        echo 0 ;;
		flock-slim)   echo 1 ;;
		flock-secure) echo 2 ;;
		binius64)    echo 3 ;;
		plonky3)     echo 4 ;;
		hashcaster)  echo 5 ;;
		plonky3-bhs) echo 6 ;;
		expander)    echo 7 ;;
		*)           echo 9 ;;
	esac
}

# ---- Summary table ----------------------------------------------------------
# Column widths chosen to fit the widest values seen (e.g. target "2^18 (262144)"
# = 13 chars, verify "1043.00 ms", peak "7043.52 MB"). Header, rule, and every
# row share $fmt; the rule's dash strings are generated at the exact field widths
# so the three always line up.
fmt='  %-11s %-8s %-14s %9s %12s %9s %11s %11s %11s\n'
dashes() { local i s=""; for ((i = 0; i < $1; i++)); do s+="-"; done; printf '%s' "$s"; }
# rule — a full-width row of dashes (under the header and between provers).
# shellcheck disable=SC2059
rule() { printf "$fmt" "$(dashes 11)" "$(dashes 8)" "$(dashes 14)" "$(dashes 9)" "$(dashes 12)" "$(dashes 9)" "$(dashes 11)" "$(dashes 11)" "$(dashes 11)"; }

# print_table TITLE ROW... — print a titled, sorted table from the given rows.
print_table() {
	local title="$1"; shift
	(( $# > 0 )) || return 0
	local row _key name target kc thr prove vfy sz mem
	echo
	echo "  $title"
	echo
	# shellcheck disable=SC2059
	printf "$fmt" "prover" "security" "target" "keccaks" "throughput" "prove" "verify" "proof size" "peak mem"
	rule
	# Sort rows by prover, then target keccaks (size) ascending. The key is
	# RE-derived here from (name, target) rather than trusting each row's embedded
	# field-1 sortkey, so cached rows written under an older order still sort right.
	# We prepend a fresh "<order><h>" key, sort, then strip it (cut -f2-).
	local SORTED=() row _n _t _h
	while IFS= read -r row; do SORTED+=("$row"); done < <(
		for row in "$@"; do
			IFS=$'\t' read -r _ _n _t _ <<< "$row"
			_h="${_t#2^}"; _h="${_h%% *}"   # "2^12 (4096)" -> "12"
			printf '%d%03d\t%s\n' "$(prover_order "$_n")" "$_h" "$row"
		done | sort | cut -f2-
	)
	local prev=""
	for row in "${SORTED[@]}"; do
		IFS=$'\t' read -r _key name target kc thr prove vfy sz mem <<< "$row"
		# Separator row between provers (rows are already grouped by prover).
		[[ -n "$prev" && "$name" != "$prev" ]] && rule
		prev="$name"
		if [[ "$thr" == "FAILED" ]]; then
			# shellcheck disable=SC2059
			printf "$fmt" "$name" "$(sec_label "$name")" "$target" "$kc" "FAILED" "$prove" "$vfy" "$sz" "$mem"
		else
			# shellcheck disable=SC2059
			printf "$fmt" "$name" "$(sec_label "$name")" "$target" "$kc" "$thr k/s" "$prove s" "$vfy" "$sz" "$mem"
		fi
	done
}

# Main table (rows 0..ST_START), then the single-threaded table (rows ST_START..).
if (( ST_START > 0 )); then
	print_table "keccak proving (${PRIMARY_THREADS} threads) — security · throughput · prove/verify · proof size · peak memory" "${ROWS[@]:0:ST_START}"
fi
if (( ${#ROWS[@]} > ST_START )); then
	print_table "single-threaded (1 thread) — flock 2^[$FLOCK_ST_SIZES], binius64 2^[$B64_ST_SIZES], plonky3 2^[$P3_ST_SIZES], hashcaster 2^[$HC_ST_SIZES]" "${ROWS[@]:ST_START}"
fi
echo
echo "  notes: 'security' = targeted provable-security bits, a config property — flock ~100"
echo "         (Johnson), plonky3 ~101 (245 queries, no PoW), hashcaster 100, binius64"
echo "         ~96 (FRI query-phase accounting). 'prove' is prover time; throughput"
echo "         = keccaks / prove. flock(prove_fast), binius64, plonky3, and hashcaster are all"
echo "         end-to-end incl. witness/trace gen (binius64 witness gen is parallelized via the"
echo "         tracked patch); verify collected for all. flock proof size = commitment + proof"
echo "         (bincode). 'keccaks' is the actual proven count, not the 2^N target (plonky3 pads"
echo "         its trace to ~1.33x; hashcaster sizes to 3·2^k = 1.5x). cached under bench-keccak-cache/ and"
echo "         reused on a plain default run;"
echo "         name a prover to re-measure it, or NO_CACHE=1 to refresh everything."
