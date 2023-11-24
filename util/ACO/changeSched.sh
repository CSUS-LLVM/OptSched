#!/usr/bin/env bash

#usage  ./modifyParam.sh SPILL_COST_FUNCTION PERP
sed -Eie "s/($1)\ .*/$1 $2/" /home/bruce/.optsched-cfg/sched.ini
echo "Changing value of $1 to $2"
