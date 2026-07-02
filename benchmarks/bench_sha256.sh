#!/usr/bin/env bash
# bench_sha256.sh — SHA-256 compression proving: Flock vs binius64, Spartan-Hyrax
# and NeutronNova, single- and multi-threaded. The SHA-256 sibling of
# bench_keccak.sh (same caching, prover-selection, two-table layout). Every prover
# SWEEPS a range of sizes and the aggregator reports its peak throughput (and the
# size at which it peaks): flock/binius64 sweep the shared 2^h list (HASH_LOG2S),
# while spartan2/neutronnova sweep their own compression-count lists.
#
# - flock:     `cargo bench --bench sha2_proof` (N independent SHA-256
#              compressions; prove_fast, incl. witness gen). Swept over HASH_LOG2S
#              up to FLOCK_MAX_LOG2 (2^16; Ligerito thrashes at 2^18 on 24 GB).
# - binius64:  `HASH=sha256 binius64/setup.sh` — N independent SHA-256
#              compressions via the tracked sha256_compressions.rs example. Swept
#              over HASH_LOG2S up to B64_MAX_LOG2 (2^14; its single-threaded
#              circuit builder is impractically slow past that), so 2^16 is
#              effectively flock-only.
# - spartan2:  microsoft/Spartan2's sha256_spartan criterion bench (T256Hyrax-
#              Engine: 256-bit curve field, Hyrax PCS). Proves ONE monolithic
#              SHA-256 R1CS over a small preimage = a chain of sequentially-
#              dependent compressions (cf. Flock sha2_chain_proof), not N
#              independent ones. Being monolithic (no folding) its throughput is
#              flat then DECLINES with size, so we sweep small preimages
#              (SPARTAN_SIZES) and report the peak. See run_spartan2.
# - neutronnova: microsoft/Spartan2's sha256_neutronnova folding bench. Folds N
#              single-block compressions, so it scales with the
#              batch; we sweep NN_SIZES and report the throughput peak. Its larger
#              sizes / ST runs are minutes each. See run_neutronnova.
#
# plonky3 (no SHA-256 AIR) and hashcaster (no SHA-256 circuit) are excluded —
# neither can prove SHA-256 out of the box.
#
# Usage: bench_sha256.sh [PROVER ...]   (no args = all). PROVER ∈
#   flock | binius64 | spartan2 | neutronnova  (aliases: nn = neutronnova,
#   both = flock+binius64, all = everything). $ONLY works too.
#   e.g.  ./bench_sha256.sh flock        ./bench_sha256.sh neutronnova
#
# Single-threaded comparison: after the main (P-core) pass, each selected prover
# is re-run single-threaded (RAYON_NUM_THREADS=1) — flock/binius64 over $ST_LOG2S,
# spartan2/neutronnova re-sweeping their own lists at 1 thread — and printed as a
# second table. ST_LOG2S="" skips the flock/binius64 single-thread sweep.
#
# Caching: every result is saved under bench-sha256-cache/<prover>_2^<h>_t<threads>.
# A plain `./bench_sha256.sh` reuses cached rows and only runs what's missing;
# naming a prover re-runs it fresh. USE_CACHE=1 reuses even with args; NO_CACHE=1
# forces fresh runs (still caching). Knobs: HASH_LOG2S (flock/binius64 MT sweep,
# default "10 11 12 13 14 16"), ST_LOG2S, SPARTAN_SIZES, NN_SIZES, FLOCK_MAX_LOG2,
# B64_MAX_LOG2, RAYON_NUM_THREADS. See ../CLAUDE.md "External Benchmarks".

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOCK_ROOT="$(cd "$BASE/.." && pwd)"

# Thread count — performance-core count by default (matches Flock's
# init_perf_thread_pool); RAYON_NUM_THREADS wins. Also part of cache keys.
if [[ -n "${RAYON_NUM_THREADS:-}" ]]; then
	THREADS="$RAYON_NUM_THREADS"
else
	if [[ "$(uname -s)" == "Darwin" ]]; then PCORES="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)"; fi
	: "${PCORES:=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
	THREADS="$PCORES"
fi
PRIMARY_THREADS="$THREADS"

HASH_LOG2S="${HASH_LOG2S:-10 11 12 13 14 16}"
# Single-thread sizes (shown in a second table); "" → skip. Defaults to 2^8..2^16,
# decoupled from the MT sweep: single-threaded runs explore where per-core
# throughput peaks (the only MT-only point dropped here is 2^18 — too slow/heavy
# single-threaded). flock runs the whole range; binius64 stops at its 2^14 cap, so
# 2^15/2^16 are effectively flock-only. (No colon: explicit "" stays empty.)
ST_LOG2S="${ST_LOG2S-8 9 10 11 12 13 14 15 16}"

# Per-prover max log2 size. A prover skips any sweep size above its cap (stderr
# notice). Raise on a bigger/faster box.
#   flock     18: ~3.4 GB @ 2^16, ~13.6 GB @ 2^18 — both fit a 24 GB box.
#   binius64  14: capped not by memory but by *circuit-construction* time — its
#                 frontend builds N SHA-256 gadgets single-threaded, which becomes
#                 impractically slow at 2^16 (65536 gadgets). 2^16/2^18 are a
#                 flock-only extension. Override with B64_MAX_LOG2=16 if you don't
#                 mind the long build (and note 2^18 ≈ 39 GB also won't fit 24 GB).
# Ligerito's larger memory footprint thrashes at 2^18 (~17.8 GB) on a 24 GB box
# — throughput collapses there — so flock is capped at 2^16 (~4.5 GB).
FLOCK_MAX_LOG2="${FLOCK_MAX_LOG2:-16}"
B64_MAX_LOG2="${B64_MAX_LOG2:-14}"

# --cooldown N (seconds between sweeps; default 0 = off; also via COOLDOWN env).
# Stripped from positional args before prover selection / the cache-read check.
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

# Prover selection (CLI args or $ONLY; default all).
do_flock=false; do_b64=false; do_spartan=false; do_nn=false
if [[ $# -gt 0 ]]; then REQUESTED=("$@")
elif [[ -n "${ONLY:-}" ]]; then
	# shellcheck disable=SC2206
	REQUESTED=($ONLY)
else REQUESTED=(all); fi
for tok in "${REQUESTED[@]}"; do
	case "$tok" in
		flock)            do_flock=true ;;
		binius64)         do_b64=true ;;
		spartan2|spartan) do_spartan=true ;;          # Spartan-Hyrax (sha256_spartan)
		neutronnova|nn)   do_nn=true ;;               # NeutronNova folding (sha256_neutronnova)
		both)             do_flock=true; do_b64=true ;;
		all)              do_flock=true; do_b64=true; do_spartan=true; do_nn=true ;;
		*) echo "unknown prover '$tok' (flock|binius64|spartan2|neutronnova|both|all)" >&2; exit 1 ;;
	esac
done

ROWS=()

# --- Result cache (mirrors bench_keccak.sh) ----------------------------------
CACHE_DIR="$BASE/bench-sha256-cache"
mkdir -p "$CACHE_DIR"
if [[ -n "${NO_CACHE:-}" ]]; then cache_read=false
elif [[ -n "${USE_CACHE:-}" ]]; then cache_read=true
elif [[ $# -eq 0 && -z "${ONLY:-}" ]]; then cache_read=true
else cache_read=false; fi

# ---- Cooldowns (opt-in via --cooldown N / COOLDOWN=N) ------------------------
# Sleep between the multi-threaded and single-threaded sweeps to let the machine
# cool; only fires when the MT pass measured something fresh (cache hits generate
# no heat). See bench_keccak.sh for the full rationale.
COOLDOWN_DID_FRESH=0
mark_fresh() { COOLDOWN_DID_FRESH=1; }
cooldown() {
	[[ "${COOLDOWN:-0}" -gt 0 && "$COOLDOWN_DID_FRESH" == 1 ]] || return 0
	echo "  cooldown: sleeping ${COOLDOWN}s before ${1:-the next sweep} to let the machine cool..." >&2
	sleep "$COOLDOWN"
	COOLDOWN_DID_FRESH=0
}

cache_file() { echo "$CACHE_DIR/${1}_2^${2}_t${THREADS}"; }
cache_lookup() {
	local f label; f="$(cache_file "$1" "$2")"
	[[ "$cache_read" == true && -s "$f" ]] || return 1
	# Size key is a log2 for flock/binius64 but a plain label ("1KiB") for
	# spartan2 — only expand 2^h when numeric (a failed $(( )) aborts the
	# whole calling function and would silently drop the row).
	if [[ "$2" =~ ^[0-9]+$ ]]; then label="2^$2 ($(( 1 << $2 )))"; else label="$2"; fi
	echo "  using cached result: $1 $label, ${THREADS}t"
	ROWS+=("$(cat "$f")")
}
add_row() {
	ROWS+=("$3")
	mark_fresh   # a fresh measurement ran (cache hits never reach add_row)
	case "$3" in
		*$'\t'FAILED$'\t'*) ;;
		*) printf '%s\n' "$3" > "$(cache_file "$1" "$2")" ;;
	esac
}

# --- formatting helpers ------------------------------------------------------
to_ms() {
	awk -v s="${1:-}" 'BEGIN {
		n = split(s, a, " "); if (n < 2) { print "n/a"; exit }
		v = a[1]; u = a[2]
		mult = (u=="s") ? 1000 : (u=="ms") ? 1 : (u=="µs"||u=="us") ? 0.001 : (u=="ns") ? 1e-6 : 0
		if (mult == 0) { print "n/a"; exit }
		printf "%.2f ms", v * mult
	}'
}

# to_secs "value unit" -> seconds as a float, or empty if unparseable.
to_secs() {
	awk -v s="${1:-}" 'BEGIN {
		n = split(s, a, " "); if (n < 2) exit
		v = a[1]; u = a[2]
		m = (u=="s") ? 1 : (u=="ms") ? 1e-3 : (u=="µs"||u=="us") ? 1e-6 : (u=="ns") ? 1e-9 : 0
		if (m > 0) printf "%.6f", v * m
	}'
}
human_size() {
	awk -v b="${1:-}" 'BEGIN {
		if (b == "") { print "n/a"; exit }
		if (b+0 < 1024)         printf "%d B", b
		else if (b+0 < 1048576) printf "%.2f KiB", b/1024
		else                    printf "%.2f MiB", b/1048576
	}'
}

# --- flock (sha2_proof) ------------------------------------------------------
# Section header in the bench output: "=== <n> compressions  (m = ...) ===".
flock_secs() {   # N LOG
	awk -v want="$1" '
		/=== .* compressions/ { for (i=1;i<=NF;i++) if ($i=="compressions") { curN=$(i-1); break } }
		curN == want && /best prove_fast:/ {
			for (i=1;i<=NF;i++) if ($i=="prove_fast:") { val=$(i+1); u=$(i+2); break }
			mult=(u=="s")?1:(u=="ms")?1e-3:(u=="µs"||u=="us")?1e-6:(u=="ns")?1e-9:0
			if (mult>0) { printf "%.6f", val*mult; exit }
		}' "$2"
}
flock_field() { # N LOG TOKEN
	awk -v want="$1" -v tok="$3" '
		/=== .* compressions/ { for (i=1;i<=NF;i++) if ($i=="compressions") { curN=$(i-1); break } }
		curN == want { for (i=1;i<=NF;i++) if ($i==tok) { print $(i+1)" "$(i+2); exit } }' "$2"
}
flock_proof_bytes() { # N LOG
	awk -v want="$1" '
		/=== .* compressions/ { for (i=1;i<=NF;i++) if ($i=="compressions") { curN=$(i-1); break } }
		curN == want && /proof size:/ { for (i=1;i<=NF;i++) if ($i=="size:") { print $(i+1); exit } }' "$2"
}

# run_flock SIZES — sweep flock over the in-budget, uncached sizes in one bench run.
run_flock() {
	local sizes="$1" log status spec h n key v thr prove vfy mem sz specs ks
	specs=(); ks=""
	for h in $sizes; do
		if (( h > FLOCK_MAX_LOG2 )); then
			echo "skipping flock 2^$h: above FLOCK_MAX_LOG2=$FLOCK_MAX_LOG2 budget." >&2
			continue
		fi
		cache_lookup flock "$h" && continue
		specs+=("$h:$(( 1 << h ))"); ks="${ks:+$ks }$h"
	done
	[[ -z "$ks" ]] && return
	echo
	echo "############### flock — sha2_proof (2^[$ks]) ###############"
	log="$(mktemp)"
	set +e
	( cd "$FLOCK_ROOT" && SHA2_LOG2S="$ks" cargo bench --bench sha2_proof ) 2>&1 | tee "$log"
	status="${PIPESTATUS[0]}"
	set -e
	if [[ "$status" -ne 0 ]]; then
		for spec in "${specs[@]}"; do h="${spec%%:*}"; n="${spec##*:}"
			add_row flock "$h" "$(printf '%03d%d' "$h" 0)"$'\t'"flock"$'\t'"2^$h ($n)"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
		done
		rm -f "$log"; return
	fi
	for spec in "${specs[@]}"; do
		h="${spec%%:*}"; n="${spec##*:}"
		key="$(printf '%03d%d' "$h" 0)"
		vfy="$(to_ms "$(flock_field "$n" "$log" "verify:")")"
		mem="$(flock_field "$n" "$log" "memory:")"; mem="${mem:-n/a}"
		sz="$(human_size "$(flock_proof_bytes "$n" "$log")")"
		v="$(flock_secs "$n" "$log")"
		if [[ -z "$v" ]]; then
			ROWS+=("$key"$'\t'"flock"$'\t'"2^$h ($n)"$'\t'"$n"$'\t'"?"$'\t'"?"$'\t'"$vfy"$'\t'"$sz"$'\t'"$mem")  # parse failed; don't cache
		else
			thr="$(awk -v v="$v" -v n="$n" 'BEGIN { printf "%.0f", n / v }')"
			prove="$(awk -v v="$v" 'BEGIN { printf "%.3f", v }')"
			add_row flock "$h" "$key"$'\t'"flock"$'\t'"2^$h ($n)"$'\t'"$n"$'\t'"$thr"$'\t'"$prove"$'\t'"$vfy"$'\t'"$sz"$'\t'"$mem"
		fi
	done
	rm -f "$log"
}

# run_b64 KEY H N — one binius64 SHA-256 compressions run (HASH=sha256 setup.sh).
run_b64() {
	local key="$1" h="$2" n="$3" log status line thr kc prove vfy sz mem
	echo
	echo "############### binius64 — sha256 2^$h ($n) ###############"
	log="$(mktemp)"
	set +e
	HASH=sha256 N_COMPRESSIONS="$n" RAYON_NUM_THREADS="$THREADS" bash "$BASE/binius64/setup.sh" 2>&1 | tee "$log"
	status="${PIPESTATUS[0]}"
	set -e
	if [[ "$status" -ne 0 ]]; then
		add_row binius64 "$h" "$key"$'\t'"binius64"$'\t'"2^$h ($n)"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
		rm -f "$log"; return
	fi
	line="$(grep -E 'throughput:' "$log" | tail -1 | sed 's/\x1b\[[0-9;]*m//g')"
	read -r thr kc prove < <(awk '{
		for (i = 1; i <= NF; i++) {
			if ($i == "throughput:")  t = $(i+1)
			if ($i == "compressions") k = $(i-1)
			if ($i == "/")            p = $(i+1)
		}
		print t, k, p
	}' <<< "$line")
	vfy="$(to_ms "$(awk '$0 ~ /verify:/ { for (i=1;i<=NF;i++) if ($i=="verify:") { print $(i+1)" "$(i+2); exit } }' "$log")")"
	mem="$(awk '/peak memory:/ { for (i=1;i<=NF;i++) if ($i=="memory:") { print $(i+1)" "$(i+2); exit } }' "$log")"
	sz="$(human_size "$(awk 'tolower($0) ~ /proof size/ { for (i=1;i<=NF;i++) if ($i=="bytes") { print $(i-1); exit } }' "$log")")"
	rm -f "$log"
	add_row binius64 "$h" "$key"$'\t'"binius64"$'\t'"2^$h ($n)"$'\t'"${kc:-?}"$'\t'"${thr:-?}"$'\t'"${prove:-?}"$'\t'"$vfy"$'\t'"$sz"$'\t'"${mem:-n/a}"
}

# run_b64_size H — run binius64 at log2 size H (honors B64_MAX_LOG2 + cache).
run_b64_size() {
	local h="$1" n
	n=$(( 1 << h ))
	if (( h > B64_MAX_LOG2 )); then
		echo "skipping binius64 2^$h: above B64_MAX_LOG2=$B64_MAX_LOG2 budget (2^18 ≈ 39 GB)." >&2
	elif cache_lookup binius64 "$h"; then :
	else run_b64 "$(printf '%03d%d' "$h" 1)" "$h" "$n"; fi
}

# --- spartan2 (microsoft/Spartan2 sha256_spartan, Hyrax engine) ---------------
# Proves ONE monolithic SHA-256 R1CS over a `bytes`-byte preimage (compressions
# = ceil((bytes+9)/64): 1 KiB->17, 2 KiB->33, 4 KiB->65, 8 KiB->129). Being
# monolithic (no folding), its throughput is flat at the small sizes and then
# DECLINES as the circuit grows, so we sweep only up to 129 compressions and the
# aggregator reports the (small-size) peak. Reported prover time is prep_prove +
# prove (witness gen + commit + proof; only the one-time `setup` is excluded),
# matching Flock's witness-gen-inclusive accounting. SPARTAN_SIZES entries are
# "bytes:exp" (exp = nearest log2 of the compression count); override to change.
SPARTAN_SIZES="${SPARTAN_SIZES:-1024:4 2048:5 4096:6 8192:7}"

# crit_time PREFIX GROUP SIZE LOG — criterion median "value unit" for
# PREFIX/GROUP/SIZE/t$THREADS (id line and time may share a line or not).
crit_time() {
	awk -v id="$1/$2/$3/t$THREADS" '
		index($0, id) { f = 1 }
		f && /time:/ {
			gsub(/[][]/, "")
			for (i = 1; i <= NF; i++) if ($i == "time:") { print $(i+3)" "$(i+4); exit }
		}' "$4"
}
spartan_time()     { crit_time spartan_sha256     "$1" "$2" "$3"; }  # Hyrax
neutronnova_time() { crit_time neutronnova_sha256 "$1" "$2" "$3"; }  # folding

# run_spartan2 — sweep sha256_spartan over SPARTAN_SIZES at $THREADS (one bench
# run covers all sizes via SP_BYTES). compressions kc = ceil((bytes+9)/64).
run_spartan2() {
	local log status spec sz_b exp kc label key prep_s prove_s secs thr prove vfy psz uncached bytes
	uncached=()
	for spec in $SPARTAN_SIZES; do
		exp="${spec##*:}"
		cache_lookup spartan2 "$exp" && continue   # label = compression exp
		uncached+=("$spec")
	done
	(( ${#uncached[@]} > 0 )) || return 0
	# Byte sizes to measure drive sha256_spartan via SP_BYTES.
	bytes="$(printf '%s,' "${uncached[@]%%:*}")"; bytes="${bytes%,}"
	echo
	echo "############### spartan2 — sha256_spartan (${bytes} B, t$THREADS) ###############"
	log="$(mktemp)"
	set +e
	SP_BYTES="$bytes" SPARTAN_BENCH=spartan SPARTAN_FILTER='^spartan_sha256/(prep_prove|prove|verify)/' BENCH_THREADS="$THREADS" \
		bash "$BASE/spartan2/setup.sh" 2>&1 | tee "$log"
	status="${PIPESTATUS[0]}"
	set -e
	for spec in "${uncached[@]}"; do
		sz_b="${spec%%:*}"; exp="${spec##*:}"; kc=$(( (sz_b + 72) / 64 )); label="$exp"
		key="$(printf '%04d2' "$sz_b")"
		if [[ "$status" -ne 0 ]]; then
			add_row spartan2 "$label" "$key"$'\t'"spartan2"$'\t'"2^$exp ($kc)"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
			continue
		fi
		# prover time = prep_prove + prove (witness gen + commit + the proof),
		# matching Flock's witness-gen-inclusive accounting.
		prep_s="$(to_secs "$(spartan_time prep_prove "$sz_b" "$log")")"
		prove_s="$(to_secs "$(spartan_time prove "$sz_b" "$log")")"
		secs="$(awk -v p="${prep_s:-}" -v q="${prove_s:-}" 'BEGIN { if (p == "" || q == "") exit; printf "%.6f", p + q }')"
		vfy="$(to_ms "$(spartan_time verify "$sz_b" "$log")")"
		psz="$(human_size "$(awk -v sz="$sz_b" '$0 ~ ("msg=" sz "B") && /proof_size=/ {
			for (i = 1; i <= NF; i++) if ($i ~ /^proof_size=/) { sub(/proof_size=/, "", $i); print $i; exit }
		}' "$log")")"
		if [[ -z "$secs" ]]; then
			ROWS+=("$key"$'\t'"spartan2"$'\t'"2^$exp ($kc)"$'\t'"$kc"$'\t'"?"$'\t'"?"$'\t'"$vfy"$'\t'"$psz"$'\t'"n/a")  # parse failed; don't cache
		else
			thr="$(awk -v v="$secs" -v n="$kc" 'BEGIN { t = n / v; printf (t < 100 ? "%.1f" : "%.0f"), t }')"
			prove="$(awk -v v="$secs" 'BEGIN { printf "%.3f", v }')"
			add_row spartan2 "$label" "$key"$'\t'"spartan2"$'\t'"2^$exp ($kc)"$'\t'"$kc"$'\t'"$thr"$'\t'"$prove"$'\t'"$vfy"$'\t'"$psz"$'\t'"n/a"
		fi
	done
	rm -f "$log"
}

# --- neutronnova (microsoft/Spartan2 sha256_neutronnova folding bench) --------
# Folds N single-block SHA-256 compressions (N = size/64 = bytes/64). Unlike
# Hyrax (stuck at its fixed 1/2 KiB preimages), NeutronNova folds an arbitrary
# number of steps, so we SWEEP its batch size and let the aggregator report the
# throughput PEAK. The default sweeps 16 -> 1024 compressions (2^4 .. 2^10): the
# curve climbs to ~512 (2^9) then regresses at 2^10, so the max lands at 512.
# Reported prover time is prep_prove + prove (matching run_spartan2 and Flock's
# witness-gen-inclusive accounting), cached one row per size under the
# `neutronnova` key. NN_SIZES entries are "bytes:exp" (exp = log2 of the
# compression count); override to narrow it, e.g. NN_SIZES="32768:9".
# NOTE: under criterion's 10-sample windows the 2^10 point (and ST runs) are
# slow -- minutes each; narrow NN_SIZES for a quick pass.
NN_SIZES="${NN_SIZES:-1024:4 2048:5 4096:6 8192:7 16384:8 32768:9 65536:10}"

run_neutronnova() {
	local log status spec sz_b exp kc label key prep_s prove_s secs thr prove vfy psz uncached bytes
	uncached=()
	for spec in $NN_SIZES; do
		exp="${spec##*:}"
		cache_lookup neutronnova "$exp" && continue   # label = compression exp
		uncached+=("$spec")
	done
	(( ${#uncached[@]} > 0 )) || return 0
	# Byte sizes to measure drive sha256_neutronnova via NN_BYTES.
	bytes="$(printf '%s,' "${uncached[@]%%:*}")"; bytes="${bytes%,}"
	echo
	echo "############### neutronnova — sha256_neutronnova (${bytes} B, t$THREADS) ###############"
	log="$(mktemp)"
	set +e
	NN_BYTES="$bytes" SPARTAN_BENCH=neutronnova SPARTAN_FILTER='^neutronnova_sha256/(prep_prove|prove|verify)/' BENCH_THREADS="$THREADS" \
		bash "$BASE/spartan2/setup.sh" 2>&1 | tee "$log"
	status="${PIPESTATUS[0]}"
	set -e
	for spec in "${uncached[@]}"; do
		sz_b="${spec%%:*}"; exp="${spec##*:}"; kc=$(( sz_b / 64 )); label="$exp"
		key="$(printf '%04d3' "$sz_b")"
		if [[ "$status" -ne 0 ]]; then
			add_row neutronnova "$label" "$key"$'\t'"neutronnova"$'\t'"2^$exp ($kc)"$'\t'"-"$'\t'"FAILED"$'\t'"-"$'\t'"-"$'\t'"-"$'\t'"-"
			continue
		fi
		# prover time = prep_prove + prove (witness gen + commit + the proof),
		# matching Flock's witness-gen-inclusive accounting.
		prep_s="$(to_secs "$(neutronnova_time prep_prove "$sz_b" "$log")")"
		prove_s="$(to_secs "$(neutronnova_time prove "$sz_b" "$log")")"
		secs="$(awk -v p="${prep_s:-}" -v q="${prove_s:-}" 'BEGIN { if (p == "" || q == "") exit; printf "%.6f", p + q }')"
		vfy="$(to_ms "$(neutronnova_time verify "$sz_b" "$log")")"
		psz="$(human_size "$(awk -v sz="$sz_b" '$0 ~ ("size=" sz "B") && /proof_size=/ {
			for (i = 1; i <= NF; i++) if ($i ~ /^proof_size=/) { sub(/proof_size=/, "", $i); print $i; exit }
		}' "$log")")"
		if [[ -z "$secs" ]]; then
			ROWS+=("$key"$'\t'"neutronnova"$'\t'"2^$exp ($kc)"$'\t'"$kc"$'\t'"?"$'\t'"?"$'\t'"$vfy"$'\t'"$psz"$'\t'"n/a")  # parse failed; don't cache
		else
			thr="$(awk -v v="$secs" -v n="$kc" 'BEGIN { t = n / v; printf (t < 100 ? "%.1f" : "%.0f"), t }')"
			prove="$(awk -v v="$secs" 'BEGIN { printf "%.3f", v }')"
			add_row neutronnova "$label" "$key"$'\t'"neutronnova"$'\t'"2^$exp ($kc)"$'\t'"$kc"$'\t'"$thr"$'\t'"$prove"$'\t'"$vfy"$'\t'"$psz"$'\t'"n/a"
		fi
	done
	rm -f "$log"
}

# sweep_provers SIZES — run selected provers over the sizes at the current threads.
# (spartan2/neutronnova sweep their own lists — SPARTAN_SIZES / NN_SIZES — so each
# runs once per pass as a single batched bench, not once per $sizes entry.)
sweep_provers() {
	local sizes="$1" h
	if [[ "$do_flock" == true ]]; then run_flock "$sizes"; fi
	for h in $sizes; do
		if [[ "$do_b64" == true ]]; then run_b64_size "$h"; fi
	done
	if [[ "$do_spartan" == true ]]; then run_spartan2; fi
	if [[ "$do_nn" == true ]]; then run_neutronnova; fi
}

# Main (multi-thread) pass, then the single-thread pass.
sweep_provers "$HASH_LOG2S"

ST_START=${#ROWS[@]}
if [[ -n "$ST_LOG2S" && "$PRIMARY_THREADS" != "1" ]]; then
	echo
	echo "=== single-threaded pass (RAYON_NUM_THREADS=1) over 2^[$ST_LOG2S] ==="
	THREADS=1
	export RAYON_NUM_THREADS=1
	cooldown "the sha256 single-threaded pass"
	sweep_provers "$ST_LOG2S"
fi

# sec_label NAME — targeted provable-security bits (config property).
sec_label() {
	case "$1" in
		flock)       echo "~100" ;;
		binius64)    echo "~96"  ;;   # FRI query-phase accounting
		spartan2)    echo "~128" ;;   # T256 curve DL (computational), Hyrax PCS
		neutronnova) echo "~128" ;;   # same T256 curve, NeutronNova folding
		*)           echo "n/a"  ;;
	esac
}

# prover_order NAME — sort rank (table groups by prover). Used by print_table to
# RE-derive the sort key from (name, target) so rows sort by prover then target
# size regardless of each row's embedded field-1 key (which is size-first / may be
# stale for cached rows).
prover_order() {
	case "$1" in
		flock)       echo 0 ;;
		binius64)    echo 1 ;;
		spartan2)    echo 2 ;;
		neutronnova) echo 3 ;;
		*)           echo 9 ;;
	esac
}

# ---- Summary table(s) -------------------------------------------------------
fmt='  %-11s %-8s %-14s %12s %12s %9s %11s %11s %11s\n'
dashes() { local i s=""; for ((i = 0; i < $1; i++)); do s+="-"; done; printf '%s' "$s"; }
# rule — a full-width row of dashes (under the header and between provers).
# shellcheck disable=SC2059
rule() { printf "$fmt" "$(dashes 11)" "$(dashes 8)" "$(dashes 14)" "$(dashes 12)" "$(dashes 12)" "$(dashes 9)" "$(dashes 11)" "$(dashes 11)" "$(dashes 11)"; }
print_table() {
	local title="$1"; shift
	(( $# > 0 )) || return 0
	local row _key name target kc thr prove vfy sz mem prev
	echo
	echo "  $title"
	echo
	# shellcheck disable=SC2059
	printf "$fmt" "prover" "security" "target" "compressions" "throughput" "prove" "verify" "proof size" "peak mem"
	rule
	# Sort by prover, then target size ascending. The key is RE-derived from
	# (name, target) — prepend a fresh "<order><h>" key, sort, strip it (cut -f2-)
	# — so cached rows whose embedded size-first key differs still sort right.
	local SORTED=() _n _t _h
	while IFS= read -r row; do SORTED+=("$row"); done < <(
		for row in "$@"; do
			IFS=$'\t' read -r _ _n _t _ <<< "$row"
			_h="${_t#2^}"; _h="${_h%% *}"   # "2^12 (4096)" -> "12"
			_h="${_h//[!0-9]/}"             # non-2^h targets ("1KiB (17)") -> digits only
			printf '%d%03d\t%s\n' "$(prover_order "$_n")" "${_h:-0}" "$row"
		done | sort | cut -f2-
	)
	prev=""
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
			printf "$fmt" "$name" "$(sec_label "$name")" "$target" "$kc" "$thr h/s" "$prove s" "$vfy" "$sz" "$mem"
		fi
	done
}

if (( ST_START > 0 )); then
	print_table "sha-256 compression proving (${PRIMARY_THREADS} threads) — security · throughput · prove/verify · proof size · peak memory" "${ROWS[@]:0:ST_START}"
fi
if (( ${#ROWS[@]} > ST_START )); then
	print_table "single-threaded (1 thread) — 2^[$ST_LOG2S]" "${ROWS[@]:ST_START}"
fi
echo
echo "  notes: flock/binius64 prove N INDEPENDENT SHA-256 compressions (compress(IV, m_i));"
echo "         spartan2 (Hyrax) proves ONE monolithic SHA-256 R1CS over a small (swept) preimage ="
echo "         a chain of sequentially-dependent compressions (cf. Flock sha2_chain_proof) — flat then"
echo "         declining throughput, so the peak is reported; neutronnova FOLDS N single-block"
echo "         compressions (swept), scaling with the batch. Neither is directly comparable to the"
echo "         independent-compression rows."
echo "         'security' = targeted provable bits (flock ~100 UDR, binius64 ~96 FRI query-phase,"
echo "         spartan2/neutronnova ~128 computational from T256 curve DL). 'prove' is prover time;"
echo "         throughput = compressions / prove, end-to-end incl. witness gen (spartan2/neutronnova:"
echo "         prep_prove + prove criterion medians — witness synthesis + commit + proof; only the"
echo "         one-time setup is excluded). flock proof size = commitment + proof (bincode);"
echo "         spartan2/neutronnova = bincode proof. Results cached under bench-sha256-cache/ and"
echo "         reused on a plain default run; name a prover to re-measure, NO_CACHE=1 to refresh."
echo "         plonky3 (no SHA-256 AIR) and hashcaster (no SHA-256 circuit) are excluded."
