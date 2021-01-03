#!/usr/bin/env python3

from gt_cost import RpGtCost
from analyze.script import *

run(RpGtCost, _0, _1, benchsuite='plaidml') \
    .keep_if(lambda blk: 'PassFinished' in blk) \
    .keep_if(lambda blk: blk.single('PassFinished')['num'] == 1)
