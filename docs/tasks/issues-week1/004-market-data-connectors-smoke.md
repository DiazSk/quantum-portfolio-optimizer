Title: Market data connectors smoke tests (Team Beta Week 1)

Description

Stabilize FX and market data flows for international feeds used by Story 4.1. Ensure connectors for the 8 target exchanges return valid data and handle rate limits/graceful degradation.

Tasks

- Verify data connectors for: LSE, Euronext, DAX, TSE, HKEX, ASX, TSX, BSE.
- Add throttling/rate-limit handling and retries.
- Create a smoke test script `tests/integration/test_market_connectors.py` that validates basic price/time data and market open/close status.

Acceptance criteria

- Connectors return valid symbols and price timestamps for a sample universe.
- Smoke tests pass locally and in CI.

Assignee: To be confirmed (Team Beta lead)
Labels: infra, data, medium-priority
Estimate: 3 SP
