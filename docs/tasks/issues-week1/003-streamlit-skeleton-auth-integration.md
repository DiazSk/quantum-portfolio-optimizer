Title: Streamlit skeleton + auth integration (Team Alpha Week 1)

Description

Create the Streamlit application skeleton for the client portal and wire authentication hooks to the existing auth system (`src/auth/multi_tenant_auth.py`). This is the foundation for Story 3.2.

Tasks

- Add `src/dashboard/client_portal.py` entrypoint with routing for pages: Dashboard, Analytics, Risk, Reports, Settings.
- Implement a lightweight `auth_hook` that calls FastAPI auth endpoints or validates JWT tokens via `MultiTenantAuthManager` (use a stub for now).
- Add a minimal Plotly sample chart on Dashboard page.
- Add unit tests for startup and auth handshake in `tests/dashboard/test_startup.py`.

Acceptance criteria

- Streamlit app starts locally and shows the Dashboard page.
- Auth handshake stub returns a valid session for a test tenant.
- A sample Plotly chart renders without errors.

Assignee: @james
Labels: feature, frontend, medium-priority
Estimate: 3 SP
