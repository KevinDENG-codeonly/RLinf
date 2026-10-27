#!/usr/bin/env bash
set -euo pipefail

if (( EUID != 0 )); then
  printf 'Run this script with sudo bash.\n' >&2
  exit 1
fi

swap_path=/swapfile-rlinf
if [[ -e "$swap_path" || -L "$swap_path" ]]; then
  printf '%s already exists; stopped without overwriting it.\n' "$swap_path" >&2
  exit 1
fi

current_swap_kib=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
if (( current_swap_kib < 2090000 || current_swap_kib > 2100000 )); then
  printf 'Expected the existing 2 GiB swap; found %s KiB. Stopped.\n' "$current_swap_kib" >&2
  exit 1
fi
if awk -v path="$swap_path" '$1 == path {found=1} END {exit !found}' /etc/fstab; then
  printf '/etc/fstab already references %s; inspect that entry first.\n' "$swap_path" >&2
  exit 1
fi

umask 077
(set -o noclobber; : > "$swap_path")
fallocate -l 30G "$swap_path"
chmod 600 "$swap_path"
mkswap "$swap_path"
swapon "$swap_path"

backup_path="/etc/fstab.backup-$(date +%Y%m%d_%H%M%S)"
cp -a /etc/fstab "$backup_path"
printf '%s none swap sw 0 0\n' "$swap_path" >> /etc/fstab
systemctl daemon-reload

printf 'fstab backup: %s\n' "$backup_path"
swapon --show
free -h
