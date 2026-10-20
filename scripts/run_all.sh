#!/bin/zsh
# Wait for the full download, then run the pipeline end to end. Logs to logs/run_all.log.
cd "$(dirname "$0")/.."
mkdir -p logs
true
echo "download finished: $(grep '^done:' data/raw/images_download_all.log)"
n=$(find data/raw/images -name '*.tif' -size -6000k | wc -l); echo "undersized files: $n"
for s in 01_segment_track 03_cell_annexin 02_fate_table 04_fate_labels 05_crops 06_baseline; do
  echo "=== $s $(date +%H:%M)"
  uv run scripts/$s.py 2>&1 | grep -v -E '^(Installed|Resolved|Downloading|Prepared|Audited|Built|Building|Uninstalled|Loading weights)|FutureWarning|binary_closing'
done
echo "=== all done $(date +%H:%M)"
