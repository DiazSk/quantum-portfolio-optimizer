#!/usr/bin/env python3
files = [
    'src/models/advanced_ensemble_pipeline.py',
    'src/models/enhanced_portfolio_optimizer.py', 
    'tests/test_story_2_1_ensemble.py'
]

total_lines = 0
print("Story 2.1 Implementation Analysis:")
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            lines = len(file.readlines())
            total_lines += lines
            print(f'{f}: {lines} lines')
    except Exception as e:
        print(f'{f}: ERROR - {e}')

print(f'\nTotal Story 2.1 Code: {total_lines} lines')
