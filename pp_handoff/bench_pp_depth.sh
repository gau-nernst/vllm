#!/usr/bin/env bash
set -euo pipefail

# PP send buffer depth A/B, one server per arm. Prefill only (output len 1):
# 100K shared prefix + 2-8K new tokens per request, Rust bench, token-ID prompts.
# 1. Warmup (discarded).
# 2. Capacity probe: closed loop at 64 in flight (engine always full); gives
#    saturated throughput per arm. The depth-0 arm's capacity C, written next to
#    the arms, sets the rates for every depth so all see the same workload.
# 3. Open-loop Poisson rates at 0.6/0.75/0.85/0.95 x C: per rate, reset the
#    prefix cache, prime the shared prefix, run ~90 s twice and keep the second
#    (JIT variants for new batch shapes land in the first). Fixed seed, so
#    prompts and arrival times are identical across arms.
# 4. One nsys window at 0.85 x C (fresh prefix cache, primed; the first ~10 s
#    ramp up, so analyze the tail).
# FIXED_CAPACITY=<req/s> pins C instead (repeat runs reuse the original rates).
# Usage: bench_pp_depth.sh <host> <log_dir> <model>  (log_dir ends in _d<depth>)
host=$1
log_dir=$2
model=$3
depth=${log_dir##*_d}
base_url="http://${host}:8000"
capacity_file="$(dirname "$log_dir")/capacity.txt"
export VLLM_USE_RUST_BENCH=1

bench() {
  local seed=$1
  local out=$2
  shift 2
  mkdir -p "$out"
  vllm bench serve \
    --backend openai \
    --base-url "$base_url" \
    --model "$model" \
    --tokenizer "$model" \
    --dataset-name random \
    --prompt-token-ids \
    --disable-shuffle \
    --ignore-eos \
    --seed "$seed" \
    --num-warmups 0 \
    --random-prefix-len 100000 \
    --random-input-len 5000 \
    --random-range-ratio 0.6 \
    --random-output-len 1 \
    --percentile-metrics ttft,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-dir "$out" \
    "$@"
}

steady_rps() {
  python -c "import json,glob,sys; r=json.load(open(glob.glob(sys.argv[1]+'/*.json')[0])); print(r['steady_state']['request_throughput'])" "$1"
}

# Fresh prefix cache, then one request to cache the shared prefix.
reset_and_prime() {
  curl -sf -X POST "$base_url/reset_prefix_cache" > /dev/null
  bench "$1" "$log_dir/bench/prime" --max-concurrency 1 --num-prompts 1 > /dev/null
}

bench 1 "$log_dir/bench/warmup" --max-concurrency 32 --num-prompts 128
bench 2 "$log_dir/bench/capacity" --max-concurrency 64 --num-prompts 320
echo "PP_DEPTH_BENCH capacity $(steady_rps "$log_dir/bench/capacity") req/s"

if [ -n "${FIXED_CAPACITY:-}" ]; then
  capacity=$FIXED_CAPACITY
else
  if [ "$depth" = 0 ]; then
    steady_rps "$log_dir/bench/capacity" > "$capacity_file"
  fi
  capacity=$(cat "$capacity_file")  # fails the arm if depth 0 did not run first
fi
echo "PP_DEPTH_BENCH rates from capacity $capacity req/s"

for frac in 0.6 0.75 0.85 0.95 1.0; do
  rate=$(python -c "print(round($frac * $capacity, 3))")
  n=$(python -c "print(max(60, round($rate * 90)))")
  for pass in a b; do
    reset_and_prime 3
    bench 3 "$log_dir/bench/rate_${frac}_${pass}" --request-rate "$rate" --num-prompts "$n"
  done
  echo "PP_DEPTH_BENCH rate $frac x C = $rate req/s done"
done

rate=$(python -c "print(round(0.85 * $capacity, 3))")
reset_and_prime 4
bench 4 "$log_dir/bench/profiled" --request-rate "$rate" --num-prompts "$(python -c "print(round($rate * 40))")" --profile
echo "PP_DEPTH_BENCH done"
