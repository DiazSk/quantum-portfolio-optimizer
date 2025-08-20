#!/usr/bin/env python
"""
Comprehensive test runner with coverage tracking
"""

import subprocess
import sys
import os
from datetime import datetime

def run_tests_with_coverage():
    """Run all tests and generate coverage report"""
    
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Define test categories
    test_suites = {
        'Unit Tests': 'tests/unit',
        'Integration Tests': 'tests/integration',
        'All Tests': 'tests'
    }
    
    results = {}
    
    for suite_name, suite_path in test_suites.items():
        if os.path.exists(suite_path):
            # Check if directory has test files
            has_tests = any(f.startswith('test_') and f.endswith('.py') 
                          for f in os.listdir(suite_path) if os.path.isfile(os.path.join(suite_path, f)))
            
            if has_tests or suite_path == 'tests':
                print(f"\n[RUNNING] {suite_name}...")
                print("-" * 40)
                
                cmd = [
                    'pytest',
                    suite_path,
                    '--cov=src',
                    '--cov-append',
                    '--cov-report=term-missing',
                    '-v'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                results[suite_name] = result.returncode == 0
                
                # Print summary
                if results[suite_name]:
                    print(f"[PASS] {suite_name} PASSED")
                else:
                    print(f"[FAIL] {suite_name} FAILED")
                    if result.stdout:
                        print(result.stdout[-500:])  # Last 500 chars of output
    
    # Generate HTML coverage report
    print("\n[GENERATING] Coverage reports...")
    subprocess.run(['coverage', 'html'], capture_output=True)
    
    # Get coverage percentage
    result = subprocess.run(['coverage', 'report'], capture_output=True, text=True)
    
    # Parse coverage percentage
    for line in result.stdout.split('\n'):
        if 'TOTAL' in line:
            parts = line.split()
            if parts:
                coverage_pct = parts[-1].rstrip('%')
                print(f"\n[COVERAGE] Total Coverage: {coverage_pct}%")
                
                try:
                    coverage_float = float(coverage_pct)
                    if coverage_float >= 70:
                        print("[SUCCESS] TARGET ACHIEVED! Coverage is above 70%!")
                    else:
                        gap = 70 - coverage_float
                        print(f"[INFO] Need {gap:.1f}% more to reach 70% target")
                except ValueError:
                    print("[WARNING] Could not parse coverage percentage")
    
    print(f"\n[REPORT] HTML report: htmlcov/index.html")
    print(f"[TIME] Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all(results.values()) if results else False

if __name__ == "__main__":
    success = run_tests_with_coverage()
    sys.exit(0 if success else 1)
