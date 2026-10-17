#!/usr/bin/env python
"""Compatibility entry point for the current R2 validation-only search.

The former six-candidate script has been retired. This wrapper runs the same
12-candidate, five-condition tuning protocol used by ``run_r2.py`` and never
selects a configuration on test data.
"""
from __future__ import annotations

import sys

from run_r2 import main


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--stage" not in argv:
        argv.extend(["--stage", "train"])
    main(argv)
