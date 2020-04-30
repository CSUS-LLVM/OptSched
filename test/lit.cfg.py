# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'OptSched'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp', '.cppm', '.m', '.mm', '.cu',
                   '.ll', '.cl', '.s', '.S', '.modulemap', '.test', '.rs']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = []

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.optsched_obj_root, 'test')

llvm_config.use_default_substitutions()

# Propagate path to symbolizer for ASan/MSan.
llvm_config.with_system_environment(
    ['ASAN_SYMBOLIZER_PATH', 'MSAN_SYMBOLIZER_PATH'])

config.substitutions.append(('%PATH%', config.environment['PATH']))
