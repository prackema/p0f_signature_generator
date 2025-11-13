#!/usr/bin/env bash
# References:
#   - https://betterdev.blog/minimal-safe-bash-script-template/
#
# Disclaimer: No AI was used and or interacted upon this script

# merge.sh — Combine first 3000 lines from multiple CSVs into one output file

trap cleanup SIGINT SIGTERM ERR EXIT

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
script_name=$(basename "${BASH_SOURCE[0]}")
lines=3000

usage() {
cat << EOF
Usage: ${script_name} -o output.csv file1.csv [file2.csv file3.csv ...]

Description:
  Takes the first 3000 lines from each input CSV file and merges them into a single output file.

Options:
  -h, --help        Show this help message and exit
  -V, --version     Print program version
  -v, --verbose     Print debug information
  -o, --output FILE Specify output CSV file name
  -l, --lines LINES Specify amount of lines to concatinate per file
EOF
    exit
}

version() {
cat << EOF
${script_name} 1.0.0
EOF
}

merge() {
 local output_file="${output}"
  > "${output_file}"

  for file in "${args[@]}"; do
    [[ ! -f "$file" ]] && msg "${YELLOW}Warning:${NOFORMAT} '$file' not found, skipping." && continue
    [[ -n "${verbose}" ]] && msg "Processing $file"
    head -n lines "$file" >> "$output_file"
  done

  msg "${GREEN}Merged CSV written to:${NOFORMAT} ${output_file}"
}

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
}

setup_colors() {
  if [[ -t 2 ]] && [[ -z "${NO_COLOR-}" ]] && [[ "${TERM-}" != "dumb" ]]; then
    NOFORMAT='\033[0m' RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
  else
    NOFORMAT='' RED='' GREEN='' YELLOW=''
  fi
}

msg() {
  echo >&2 -e "${script_name}: ${1-}"
}

die() {
  local msg=$1
  local code=${2-1}
  msg "$msg"
  exit "$code"
}

help() {
  local msg=$1
  die "${msg}\nTry '${script_name} --help' for more information."
}

parse_params() {
  while :; do
    case "${1-}" in
      -h | --help) usage ;;
      -V | --version) version; exit ;;
      -v | --verbose) verbose=1 ;;
      -o | --output)
        output="${2-}"
        [[ -z "${2-}" ]] && help "Option requires an argument -- '${1}'"
        shift
        ;;
      -l | --lines)
        lines="${2-}"
        [[ -z "${2-}" ]] && help "Option requires an argument -- '${1}'"
        shift
        ;;
      -?*) help "Unknown option: $1" ;;
      *) break ;;
    esac
    shift
  done

  args=("$@")
  [[ -z "${output-}" ]] && help "Missing required parameter: --output"
  [[ ${#args[@]} -eq 0 ]] && help "Missing input CSV files"
}

parse_params "$@"
setup_colors
merge
