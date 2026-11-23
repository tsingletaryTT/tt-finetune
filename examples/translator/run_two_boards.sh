#!/bin/bash
# Two tt-finetune jobs at once on a 4-chip host with two p300c boards: one job per board.
# Run it under a 4-chip lease:  gozer run --chips 4 --who ... --reason ... -- ./run_two_boards.sh OUTDIR [STEPS]
#
# Three things this script works around (found 2026-11-23, tt-finetune v0.2.1):
#  1. A job must see BOTH chips of its board. Pinning one chip makes tt-metal report a CUSTOM cluster
#     and ask for a fabric mesh graph descriptor.
#  2. The PJRT plugin writes /tmp/tt_pjrt_system_descriptor (tempfile.gettempdir()). Two jobs share it,
#     and the second fails at checkpoint export with "Error code: 13". A private TMPDIR per job fixes it.
#     Keep the TMPDIR path short: Unix socket paths are limited to about 107 characters.
#  3. train.py passes a per-job workspace=, because the default manifest path is shared too.
OUT=${1:?usage: run_two_boards.sh OUTDIR [STEPS]}; STEPS=${2:-300}
PY=${PYTHON:-python}; HERE=$(cd "$(dirname "$0")" && pwd)
VIS=("0000:01:00.0,0000:02:00.0" "0000:03:00.0,0000:04:00.0")   # adjust to your boards (tt-smi -ls)
CFG=("16 2e-4" "32 2e-4")                                        # RANK LR for each job
for i in 0 1; do
  set -- ${CFG[$i]}; mkdir -p "$OUT/job$i" "/tmp/ttft-j$i"
  ( cd "$OUT/job$i" && TMPDIR=/tmp/ttft-j$i TT_VISIBLE_DEVICES=${VIS[$i]} \
      timeout 3600 $PY "$HERE/train.py" "$OUT/job$i" "$STEPS" "$1" "$2" > train.log 2>&1; echo "exit=$?" >> train.log ) &
done
wait
