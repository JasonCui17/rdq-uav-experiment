#!/usr/bin/env python3
"""Candidate-export convenience entry using the shared evaluator implementation."""
import sys
from evaluate_lidar_uav_v2 import main
if __name__=="__main__":
    if "--export-candidates" not in sys.argv: sys.argv.append("--export-candidates")
    main()
