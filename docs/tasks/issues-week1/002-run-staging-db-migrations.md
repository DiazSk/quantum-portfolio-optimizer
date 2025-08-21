Title: Run DB migrations in staging and smoke-test tenant isolation

Description

Execute the DB migration plan referenced in the sprint doc on the staging environment and run smoke tests to validate tenant isolation and role mapping.

Steps

1. Back up staging DB snapshot.
2. Apply migrations:
   - `src/database/migrations/003_multi_tenant_auth.sql`
   - `src/database/migrations/add_portal_tables.sql`
   - Any SQL snippets referenced in `docs/stories/4.2.alternative-asset-integration.story.md`
3. Run smoke tests for tenant creation, user role mapping, and API auth flows.
4. Roll back if any critical failures are observed.

Acceptance criteria

- Migrations applied without data corruption.
- Tenant isolation smoke tests pass.
- Authentication and role mappings verified for sample tenants.

Assignee: @james
Labels: ops, db, high-priority
Estimate: 2 SP
