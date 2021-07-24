#!/usr/bin/env bash

# Given a single file "$1.*.log" containing the stdout from several runs,
# extracts the generated CPU2017.*.log files and copies them to a subdirectory "$1.logs/"
set -e
mkdir -p "$1.logs/"
grep 'The log for this run is in' "$1".*.log | sed -E 's/^.*(CPU.*\.log).*$/\1/g' | xargs -I {} cp /home/cpu2017/result/{} "$1.logs/"
(cd "$1.logs"; grep '  Building' * | sed -E 's/^(\S*\.log):  Building ([^ ]*).*$/\1 \2.log/' | xargs -n2 mv)
rm "$1".logs/CPU*
