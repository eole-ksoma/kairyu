#!/bin/sh
set -eu
exec python3 "$(dirname "$0")/control.py" "${1:-up}"
