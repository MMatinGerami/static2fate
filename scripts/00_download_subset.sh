#!/bin/zsh
# Download S-BIAD2515 (CC0) from the EBI FTP mirror: the wells given as arguments, or all 30.
# Per well: 4 FOVs, T0001..T0013 live frames x 4 channels, T0014 terminal (C01 Hoechst, C05 AnnexinV).
# First run (19 Oct) fetched C-02 C-07 C-08 C-10; complete files are skipped on rerun.
set -u
BASE="https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/515/S-BIAD2515/Files"
OUT="$(dirname "$0")/../data/raw/images"
mkdir -p "$OUT"
list=$(mktemp)
WELLS=("$@")
if [ ${#WELLS[@]} -eq 0 ]; then
  WELLS=(); for r in C D E; do for c in $(seq -f "%02g" 2 11); do WELLS+=("${r}-${c}"); done; done
fi
for w in "${WELLS[@]}"; do
  for f in 1 2 3 4; do
    for t in $(seq -f "%04g" 1 13); do
      for c in 1 2 3 4; do echo "${w}_F000${f}_T${t}_Z0001_C0${c}.tif"; done
    done
    echo "${w}_F000${f}_T0014_Z0001_C01.tif"
    echo "${w}_F000${f}_T0014_Z0001_C05.tif"
  done
done > "$list"
echo "$(wc -l < "$list") files to fetch into $OUT"
cd "$OUT"
# 6 parallel curls, resume partial files, skip complete ones
xargs -P 6 -I{} sh -c 'f="{}"; [ -s "$f" ] && [ "$(stat -f%z "$f")" -gt 7000000 ] && exit 0; curl -sS -C - --retry 5 --retry-delay 5 -o "$f" "'"$BASE"'/$f" || echo "FAILED $f"' < "$list"
echo "done: $(ls *.tif | wc -l) files, $(du -sh . | cut -f1)"
