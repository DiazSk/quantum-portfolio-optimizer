# Week 2 Sprint Plan - Epic 3 & 4 Development

**Sprint Period**: Week 2 (August 27 - September 2, 2025)  
**Team Alpha Lead**: James  
**Team Beta Lead**: TBC (assigned this week)  
**Scrum Master**: Bob  

---

## Sprint Objectives

### Primary Goals
- **Team Alpha**: Implement core Streamlit pages with real portfolio data integration
- **Team Beta**: Establish Team Beta lead and begin GlobalPortfolioOptimizer integration
- **Infrastructure**: Execute staging DB migrations and establish CI/CD pipeline

### Week 2 Deliverable Target
**End-to-end API auth + dashboard auth flow; integration tests passing**

---

## Team Alpha Tasks (James)

### Task 2.1: Core Streamlit Pages Implementation
**Priority**: High | **Estimate**: 5 SP

**Scope**: Implement Dashboard and Analytics pages with real data integration
- Enhance Dashboard page with live portfolio data from optimization engine
- Create Analytics page with performance attribution and risk metrics
- Implement Plotly charts with real data (not stubs)
- Add portfolio performance tracking and comparison features

**Acceptance Criteria**:
- Dashboard displays real portfolio holdings and performance
- Analytics page shows risk metrics, factor exposure, and attribution
- Plotly charts render portfolio data without errors
- Charts update with tenant-specific data based on auth context

**Files to Create/Modify**:
- `src/dashboard/pages/dashboard.py` - Enhanced dashboard implementation
- `src/dashboard/pages/analytics.py` - Portfolio analytics and risk metrics
- `src/dashboard/services/portfolio_service.py` - Data service integration
- `tests/dashboard/test_portfolio_integration.py` - Integration tests

---

### Task 2.2: Role-Based Menu System
**Priority**: Medium | **Estimate**: 3 SP

**Scope**: Implement dynamic navigation based on user roles from auth system
- Connect role-based navigation to real auth system responses
- Implement menu filtering for different user types (viewer/analyst/admin)
- Add permission-based feature access control
- Create tenant-specific branding and customization

**Acceptance Criteria**:
- Navigation menus filter based on authenticated user role
- Features are hidden/shown based on user permissions
- Tenant branding applies correctly in multi-tenant context
- Role changes reflect immediately in UI without restart

**Files to Create/Modify**:
- `src/dashboard/components/navigation.py` - Role-based menu system
- `src/dashboard/auth_hook.py` - Enhanced role handling
- `src/dashboard/styles/tenant_themes.py` - Multi-tenant branding
- `tests/dashboard/test_role_based_access.py` - Role-based tests

---

### Task 2.3: Staging Database Migration Execution
**Priority**: High | **Estimate**: 2 SP

**Scope**: Execute database migrations in staging environment
- Run `python scripts/run_staging_migrations.py --apply` 
- Validate tenant isolation and role mapping
- Test multi-tenant authentication flows
- Document any migration issues and resolutions

**Acceptance Criteria**:
- All migrations complete successfully in staging
- Tenant isolation verified through testing
- Auth system can create and authenticate users
- Rollback procedures tested and documented

**Environment Requirements**:
- Staging database access configured
- STAGING_DB_DSN environment variable set
- Database backup completed before migration

---

## Team Beta Tasks (TBC Lead)

### Task 2.4: Team Beta Lead Assignment
**Priority**: Critical | **Estimate**: 0 SP (administrative)

**Scope**: Confirm Team Beta lead and accept Story 4.2 handoff
- Assign Team Beta lead (internal team member or external consultant)
- Review Story 4.2 alternative asset integration requirements
- Accept handoff of global markets infrastructure from Story 4.1
- Plan Team Beta sprint capacity and timeline

**Acceptance Criteria**:
- Team Beta lead identified and confirmed
- Story 4.2 requirements reviewed and accepted
- Team capacity confirmed for 16 SP story implementation
- Week 2-4 timeline agreed upon

---

### Task 2.5: GlobalPortfolioOptimizer Integration
**Priority**: High | **Estimate**: 5 SP

**Scope**: Connect real portfolio optimization engine to international markets
- Integrate Story 4.1 global markets data with optimization engine
- Implement 8 international market feeds integration
- Test portfolio optimization with real international securities
- Validate multi-currency portfolio construction

**Acceptance Criteria**:
- GlobalPortfolioOptimizer accepts international market data
- 8 target exchanges provide data to optimization engine
- Multi-currency portfolio optimization works end-to-end
- Performance meets targets (<30 seconds for 8+ international securities)

**Files to Create/Modify**:
- `src/portfolio/global_optimizer_integration.py` - Integration layer
- `src/data/international_feeds.py` - Market data aggregation
- `tests/integration/test_global_optimization.py` - End-to-end tests

---

## Infrastructure & DevOps Tasks

### Task 2.6: CI/CD Pipeline Enhancement
**Priority**: Medium | **Estimate**: 2 SP

**Scope**: Enhance CI pipeline for multi-component testing
- Add Streamlit app testing to CI pipeline
- Configure environment-based test execution
- Add integration test stage for auth flows
- Set up staging deployment automation

**Acceptance Criteria**:
- CI pipeline runs Streamlit tests successfully
- Auth integration tests run in CI environment
- Staging deployment automated and tested
- Pipeline provides clear feedback on failures

---

## Sprint Timeline

### Monday (Day 1)
- **09:00**: Sprint planning meeting with Team Alpha & Beta leads
- **10:00**: James begins Task 2.1 (Dashboard implementation)
- **14:00**: Team Beta lead confirmation meeting
- **EOD**: Task 2.3 staging migration executed

### Tuesday-Wednesday (Days 2-3)
- **Team Alpha**: Core Streamlit pages development
- **Team Beta**: Global optimizer integration setup
- **Daily standups**: 09:00 progress check and blocker identification

### Thursday (Day 4)
- **Team Alpha**: Role-based menu implementation
- **Team Beta**: International market feeds testing
- **Mid-week demo**: Internal progress demonstration

### Friday (Day 5)
- **Integration testing**: End-to-end auth + dashboard flow
- **Sprint review**: Week 2 deliverables validation
- **Retrospective**: Process improvements for Week 3

---

## Definition of Done - Week 2

### Technical Requirements
- [ ] Dashboard page displays real portfolio data
- [ ] Analytics page shows risk metrics and attribution
- [ ] Role-based navigation works with real auth system
- [ ] Staging database migrations completed successfully
- [ ] Team Beta lead confirmed and Story 4.2 accepted
- [ ] Global portfolio optimizer integrates with international markets
- [ ] CI/CD pipeline enhanced for multi-component testing

### Quality Gates
- [ ] All unit tests pass (coverage ≥ 85%)
- [ ] Integration tests for auth flow pass
- [ ] Streamlit app starts and authenticates users
- [ ] Performance targets met (dashboard < 500ms)
- [ ] No critical security vulnerabilities identified

### Documentation
- [ ] Task completion summaries documented
- [ ] Migration results and lessons learned documented
- [ ] Team Beta handoff documentation complete
- [ ] Week 3 planning input prepared

---

## Risk Mitigation

### Identified Risks
1. **Team Beta lead assignment delay** → Impact: Story 4.2 timeline
2. **Staging migration issues** → Impact: Auth integration testing
3. **Integration complexity** → Impact: End-to-end flow delivery

### Mitigation Strategies
1. **Backup Team Beta lead identified** (assign by Tuesday if primary unavailable)
2. **Migration rollback procedures ready** (tested in Week 1)
3. **Incremental integration approach** (auth first, then portfolio data)

---

## Success Metrics

### Team Alpha (James)
- Dashboard and Analytics pages functional with real data
- Role-based navigation implemented and tested
- Staging migrations completed successfully

### Team Beta (TBC)
- Team lead confirmed and Story 4.2 planning complete
- Global optimizer integration functional
- International market feeds operational

### Sprint Overall
- End-to-end auth + dashboard flow demonstrated
- Integration tests passing in CI
- Week 3 readiness confirmed

---

## Next Sprint Preview

**Week 3 Focus**: Story 3.2 polishing and Story 4.2 alternative assets implementation
- Team Alpha: UX polishing, export features, alert system
- Team Beta: Alternative asset data models and collectors
- Target: User acceptance demo preparation

---

**Sprint prepared by**: Bob (Scrum Master)  
**For questions**: Contact James (Team Alpha) or sprint team leads  
**Sprint kickoff**: Monday 09:00 (all team leads required)
