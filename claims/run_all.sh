#!/usr/bin/env sh
# Run every claim in sequence. Pass --smoke (about 1.4 h, pipeline check only),
# --quick (about 4.9 h, the recommended evaluation path) or --full (about 63 h).
# Times are for the reference GTX 1650; see README.txt.
set -eu
[ $# -ge 1 ] || { echo "usage: $0 --smoke|--quick|--full [extra args]"; exit 2; }
D="$(cd "$(dirname "$0")" && pwd)"
FAILED=""
for c in claim1_main_defense claim2_ablation claim3_density_cliff \
         claim4_cross_dataset claim5_adaptive claim6_probe_evasion; do
  echo
  echo "################ $c ################"
  sh "$D/$c/run.sh" "$@" || FAILED="$FAILED $c"
done
echo
if [ -n "$FAILED" ]; then
  echo "claims with failing assertions:$FAILED"
  exit 1
fi
echo "all claims passed"
