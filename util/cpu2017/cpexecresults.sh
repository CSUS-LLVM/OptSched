#!/usr/bin/env bash
grep -E '\.(csv|rsf|txt)' "$@" | # Find every .csv, .rsf, or .txt mentioned in the logs
    sed -E 's/^\s*format:.*-> (.*)$/\1/g' | # Extract the filenames for that
    xargs -L1 -I {} cp {} . # Take each file and copy it to the current directory
