#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "usage: $0 <t2v|i2v> <output-directory>" >&2
  exit 2
}

case "$1" in
  t2v) expected=b27ee7ef5434602feaa9bcb3b9474ccb6b7fe323ce812af0247bc6d234795345 ;;
  i2v) expected=c08e297082638ce11afa83eda5021b0e157f65906b1982a34df16f9d71a45352 ;;
  *) echo "unknown workload: $1" >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_file=$script_dir/$1.csv
actual=$(sha256sum "$source_file" | awk '{print $1}')
[[ $actual == "$expected" ]] || {
  echo "SHA-256 mismatch: expected $expected, got $actual" >&2
  exit 1
}

mkdir -p "$2"
for rank in {0..7}; do
  install -m 0644 "$source_file" "$2/tunableop-results${rank}.csv"
done

echo "PYTORCH_TUNABLEOP_FILENAME=$2/tunableop-results.csv"
echo "SHA-256=$actual"
