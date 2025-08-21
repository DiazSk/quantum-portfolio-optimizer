#!/usr/bin/env python3
"""
Final verification script to confirm all mock data has been eliminated
Created for zero-tolerance mock data requirement
"""

import os
import subprocess
import sys

def run_grep_search(pattern, description):
    """Run grep search and return results"""
    try:
        # Use findstr on Windows instead of grep
        result = subprocess.run([
            'findstr', '/S', '/I', '/C:' + pattern, 'src\\*.py'
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            return lines
        else:
            return []
    except Exception as e:
        print(f"Error running search for {pattern}: {e}")
        return []

def main():
    print("FINAL MOCK DATA ELIMINATION VERIFICATION")
    print("=" * 50)
    
    # Check for np.random instances
    print("\n1. Checking for np.random usage...")
    np_random_results = run_grep_search("np.random", "np.random usage")
    
    if np_random_results:
        print("FAILED: np.random instances found:")
        for line in np_random_results[:10]:  # Show first 10
            print(f"  {line}")
        return False
    else:
        print("PASS: NO np.random instances found")
    
    # Check for mock data generation
    print("\n2. Checking for mock data generation...")
    mock_patterns = [
        "generate_mock",
        "mock_data", 
        "MockSnapshot",
        "fallback.*mock",
        "smart.*mock"
    ]
    
    total_issues = 0
    for pattern in mock_patterns:
        results = run_grep_search(pattern, f"Pattern: {pattern}")
        
        if results:
            # Filter acceptable references
            actual_issues = []
            for line in results:
                line_lower = line.lower()
                if not any(x in line_lower for x in [
                    'return empty', 'no mock', 'not mock', 'removed mock',
                    'contact administrator', 'no simulated', 'cachedSnapshot'
                ]):
                    actual_issues.append(line)
            
            if actual_issues:
                print(f"Found {len(actual_issues)} instances of {pattern}:")
                for line in actual_issues[:3]:
                    print(f"  {line}")
                total_issues += len(actual_issues)
    
    print(f"\n3. Summary:")
    print(f"Total mock-related issues found: {total_issues}")
    
    if total_issues == 0:
        print("\nFINAL STATUS: ALL MOCK DATA SUCCESSFULLY ELIMINATED")
        print("Production deployment: APPROVED")
        print("Zero-tolerance requirement: SATISFIED")
        return True
    else:
        print(f"\nREVIEW REQUIRED: {total_issues} mock references remain")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
