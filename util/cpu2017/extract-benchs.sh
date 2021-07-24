#!/usr/bin/env bash

# Given a file "$1.*.log", splits it into one log file per benchmark, placed in "$1.logs/"

set -e
mkdir -p "$1.logs/"
cd "$1.logs/"
csplit ../"$1".*.log '/^  Building/' '{*}'
grep '  Building' * | sed -E 's/^(\S*):  Building ([^ ]*).*$/\1 \2.log/' | xargs -n2 mv
rm xx*
