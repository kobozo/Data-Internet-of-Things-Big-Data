#!/usr/bin/env bash
# =====================================================================
# clear.sh  --  reset the data folder to a clean state.
#
# Run between demo recordings so the screencast starts with an empty
# data/ folder and the generated CSVs / events DB / plots are fresh.
#
# Usage:
#   ./clear.sh           # delete the runtime data
#   ./clear.sh --dry-run # show what would be deleted, don't touch
#   ./clear.sh --yes     # skip confirmation prompt
# =====================================================================
set -euo pipefail

# resolve the project root regardless of where the script is invoked from
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR="$SCRIPT_DIR/data"

# ----- colour helpers ------------------------------------------------
if [ -t 1 ]; then
  RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'
  BOLD='\033[1m';  DIM='\033[2m';    RESET='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BOLD=''; DIM=''; RESET=''
fi

# ----- args ----------------------------------------------------------
DRY_RUN=false
ASSUME_YES=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --yes|-y)  ASSUME_YES=true ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo -e "${RED}unknown arg: $arg${RESET}" >&2
      exit 2 ;;
  esac
done

# ----- what to delete -----------------------------------------------
# files inside data/ EXCEPT the .gitkeep placeholder
# plus the ultralytics `runs/` cache at project root.
TARGETS=(
  "$DATA_DIR/frames.csv"
  "$DATA_DIR/tracks.csv"
  "$DATA_DIR/events.sqlite"
  "$DATA_DIR/events.sqlite-journal"
  "$DATA_DIR/events.sqlite-shm"
  "$DATA_DIR/events.sqlite-wal"
  "$DATA_DIR/report"
  "$DATA_DIR/_test_run"
  "$SCRIPT_DIR/runs"
)

# ----- preview ------------------------------------------------------
echo -e "${BOLD}clear.sh${RESET}  (project: ${DIM}$SCRIPT_DIR${RESET})"
echo
existing=()
for t in "${TARGETS[@]}"; do
  if [ -e "$t" ]; then
    if [ -d "$t" ]; then
      size=$(du -sh "$t" 2>/dev/null | awk '{print $1}')
      echo -e "  ${YELLOW}dir  ${RESET}${t/$SCRIPT_DIR\//}   ${DIM}($size)${RESET}"
    else
      size=$(du -h "$t" 2>/dev/null | awk '{print $1}')
      echo -e "  ${YELLOW}file ${RESET}${t/$SCRIPT_DIR\//}   ${DIM}($size)${RESET}"
    fi
    existing+=("$t")
  fi
done

if [ "${#existing[@]}" -eq 0 ]; then
  echo -e "  ${GREEN}data folder already clean.${RESET}"
  exit 0
fi
echo

if $DRY_RUN; then
  echo -e "${DIM}--dry-run set; nothing deleted.${RESET}"
  exit 0
fi

# ----- confirm ------------------------------------------------------
if ! $ASSUME_YES; then
  read -r -p "Delete these? [y/N] " ans
  case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "aborted."; exit 1 ;;
  esac
fi

# ----- delete -------------------------------------------------------
for t in "${existing[@]}"; do
  rm -rf -- "$t"
done

# make sure the data dir + .gitkeep are still there
mkdir -p "$DATA_DIR"
[ -e "$DATA_DIR/.gitkeep" ] || touch "$DATA_DIR/.gitkeep"

echo -e "${GREEN}cleared.${RESET}"
