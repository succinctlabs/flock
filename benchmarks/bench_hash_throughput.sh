#!/usr/bin/env bash
# Regenerate the README's hash proving-throughput matrix.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG2S="${LOG2S:-10 12 14}"
RUNS="${RUNS:-3}"

if [[ -z "${MT_THREADS:-}" ]]; then
	case "$(uname -s)" in
		Darwin)
			MT_THREADS="$(sysctl -n hw.perflevel0.physicalcpu)"
			;;
		Linux)
			MT_THREADS="$(lscpu -p=SOCKET,CORE | awk -F, '!/^#/ { print $1 "," $2 }' | sort -u | wc -l)"
			;;
		*)
			MT_THREADS="$(getconf _NPROCESSORS_ONLN)"
			;;
	esac
fi

[[ "$MT_THREADS" =~ ^[1-9][0-9]*$ ]] || {
	echo "MT_THREADS must be a positive integer, got '$MT_THREADS'" >&2
	exit 1
}
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || {
	echo "RUNS must be a positive integer, got '$RUNS'" >&2
	exit 1
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
results="$work/results.tsv"

run_bench() {
	local threads="$1"
	echo
	echo "=== hash throughput: ${threads} thread(s), batches 2^[${LOG2S}] ==="
	(
		cd "$ROOT"
		RAYON_NUM_THREADS="$threads" \
			HASH_BENCH_LOG2S="$LOG2S" \
			HASH_BENCH_RUNS="$RUNS" \
			cargo bench --bench hash_throughput
	) 2>&1 | tee "$work/run-${threads}.log"
	awk -F '\t' '$1 == "RESULT"' "$work/run-${threads}.log" >> "$results"
}

run_bench "$MT_THREADS"
if [[ "$MT_THREADS" != "1" ]]; then
	run_bench 1
fi

lookup() {
	local hash="$1" layout="$2" batch="$3" threads="$4"
	awk -F '\t' \
		-v hash="$hash" -v layout="$layout" -v batch="$batch" -v threads="$threads" \
		'$1 == "RESULT" && $2 == hash && $3 == layout && $4 == batch && $5 == threads {
			found = 1
			printf "%.1f", $7
			exit
		}
		END { if (!found) exit 1 }' "$results"
}

echo
echo "| Hash | Batch | 1T row-major | 1T batch-major | ${MT_THREADS}T row-major | ${MT_THREADS}T batch-major |"
echo "|---|---:|---:|---:|---:|---:|"
for hash_spec in "sha2:SHA-256" "blake3:BLAKE3" "keccak:Keccak-f[1600]"; do
	hash="${hash_spec%%:*}"
	label="${hash_spec#*:}"
	for log2 in $LOG2S; do
		batch="$((1 << log2))"
		st_row="$(lookup "$hash" row-major "$batch" 1)"
		st_batch="$(lookup "$hash" batch-major "$batch" 1)"
		mt_row="$(lookup "$hash" row-major "$batch" "$MT_THREADS")"
		mt_batch="$(lookup "$hash" batch-major "$batch" "$MT_THREADS")"
		printf '| %s | %s | %s | %s | %s | %s |\n' \
			"$label" "$batch" "$st_row" "$st_batch" "$mt_row" "$mt_batch"
	done
done
