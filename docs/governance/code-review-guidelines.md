# Code Review Guidelines

## Overview
This document establishes code review standards and processes for the async-toolformer-orchestrator project to ensure code quality, security, and knowledge sharing.

## Review Requirements

### Mandatory Reviews
All changes require at least **1 approval** from a code owner before merging:

| File/Directory | Required Reviewers | Rationale |
|----------------|-------------------|-----------|
| `/src/` | Architecture team | Core functionality impacts |
| `/docs/architecture/` | Architecture team consensus | Design decisions |
| `/security/`, `/docs/compliance/` | Security team | Security implications |
| `/k8s/`, `/docker-compose*.yml` | DevOps + Platform teams | Infrastructure changes |
| `/.github/workflows/` | DevOps team | CI/CD pipeline changes |
| `/scripts/performance/` | Performance + Architecture teams | Performance impacts |

### Auto-Approval Exceptions
These changes can be merged with single approval:
- Documentation updates (non-architectural)
- Test file additions/updates
- Minor configuration tweaks
- Dependency version bumps (non-breaking)

## Review Standards

### Code Quality Checklist

#### Functionality
- [ ] Code accomplishes the stated purpose
- [ ] Edge cases are handled appropriately
- [ ] Error handling is comprehensive and appropriate
- [ ] Performance implications are acceptable
- [ ] Memory usage is reasonable

#### Code Style
- [ ] Follows project style guidelines (ruff, mypy)
- [ ] Variable and function names are descriptive
- [ ] Code is self-documenting or well-commented
- [ ] No commented-out code blocks
- [ ] Imports are organized and minimal

#### Testing
- [ ] New functionality has appropriate test coverage
- [ ] Tests are meaningful and test the right things
- [ ] Test names are descriptive
- [ ] No flaky or unreliable tests introduced
- [ ] Performance tests included for performance-critical changes

#### Security
- [ ] No secrets or credentials in code
- [ ] Input validation is appropriate
- [ ] No SQL injection or similar vulnerabilities
- [ ] Authentication and authorization handled correctly
- [ ] Dependencies are from trusted sources

#### Documentation
- [ ] Public APIs are documented
- [ ] Complex logic is explained
- [ ] Breaking changes are noted
- [ ] Migration guides provided if needed
- [ ] Architecture decisions are recorded (ADRs)

### Performance Review Checklist

#### For Performance-Critical Changes
- [ ] Benchmarks show acceptable performance
- [ ] Memory usage profiled and acceptable
- [ ] No obvious performance anti-patterns
- [ ] Parallel execution efficiency maintained
- [ ] Rate limiting not negatively impacted

#### Scalability Concerns
- [ ] Code scales with increased load
- [ ] No N+1 query patterns or similar
- [ ] Caching strategy is appropriate
- [ ] Resource cleanup is proper
- [ ] Graceful degradation under load

### Security Review Checklist

#### For All Changes
- [ ] No hardcoded secrets or credentials
- [ ] Input sanitization is appropriate
- [ ] Output encoding prevents injection
- [ ] Error messages don't leak sensitive info
- [ ] Logging doesn't expose sensitive data

#### For High-Risk Changes
- [ ] Threat modeling completed
- [ ] Security testing included
- [ ] Penetration testing considered
- [ ] Compliance requirements met
- [ ] Third-party dependencies audited

## Review Process

### 1. Pre-Review (Author)
- [ ] Self-review completed
- [ ] Tests pass locally
- [ ] Pre-commit hooks pass
- [ ] Branch is up-to-date with main
- [ ] PR description is complete and clear

### 2. Review Assignment
- **Automatic**: GitHub assigns based on CODEOWNERS
- **Manual**: Author can request specific reviewers
- **Escalation**: Complex changes may need architecture review

### 3. Review Execution
- **Timeframe**: 24 hours for normal PRs, 4 hours for hotfixes
- **Depth**: Proportional to change impact and risk
- **Feedback**: Constructive, specific, and actionable

### 4. Review Resolution
- **Approval**: LGTM with optional minor suggestions
- **Request Changes**: Must be addressed before merge
- **Comments**: Non-blocking suggestions for improvement

## Review Types

### 1. Code Review (Standard)
- Focus on functionality, style, and maintainability
- All non-trivial changes require this review
- Automated checks must pass first

### 2. Architecture Review
- Required for significant design changes
- Must include ADR documentation
- Consensus required from architecture team
- Performance implications considered

### 3. Security Review
- Required for security-sensitive changes
- Includes threat modeling
- May require security team consensus
- Compliance implications assessed

### 4. Performance Review  
- Required for performance-critical paths
- Includes benchmark results
- Memory usage profiling
- Scalability assessment

## Review Comments

### Comment Categories
Use these prefixes for clarity:

- **MUST**: Blocking issue that must be fixed
- **SHOULD**: Strong suggestion for improvement  
- **CONSIDER**: Optional suggestion for consideration
- **QUESTION**: Clarification needed
- **NITPICK**: Minor style/preference comment

### Example Comments
```
MUST: This introduces a SQL injection vulnerability. Please use parameterized queries.

SHOULD: Consider adding error handling for the network timeout case.

CONSIDER: This could be more efficient using a dictionary lookup instead of linear search.

QUESTION: What happens if the API returns an unexpected response format?

NITPICK: Minor style - consider using f-strings instead of .format()
```

### Response Guidelines
- **Address all MUST items** before requesting re-review
- **Acknowledge SHOULD items** and implement or explain why not
- **Thank reviewers** for their time and feedback
- **Ask for clarification** if comments are unclear

## Reviewer Guidelines

### What to Review
1. **Correctness**: Does the code do what it claims?
2. **Design**: Is the approach sound and maintainable?
3. **Complexity**: Is the code as simple as possible?
4. **Testing**: Are there adequate tests?
5. **Documentation**: Is it clear what the code does?

### What NOT to Review
- Personal style preferences (use automated tools)
- Micro-optimizations (unless performance-critical)
- Existing code not touched by the PR
- Implementation details when the approach is sound

### Review Best Practices
- **Be constructive**: Suggest improvements, don't just point out problems
- **Be specific**: Give examples of what you mean
- **Be timely**: Review within the expected timeframe
- **Be thorough**: Don't rubber-stamp reviews
- **Be respectful**: Critique code, not people

## Escalation Process

### When to Escalate
- Fundamental disagreement on approach
- Security concerns require specialist input
- Performance implications need deeper analysis
- Architecture decisions need broader consensus

### Escalation Path
1. **Team Discussion**: Discuss in team chat/standup
2. **Architecture Review**: Schedule with architecture team
3. **Technical Lead**: Escalate to technical lead
4. **Engineering Manager**: Final technical arbitration

## Metrics and Monitoring

### Review Quality Metrics
- Average time to first review
- Review cycle time (start to merge)
- Number of review rounds per PR
- Post-merge bug rate by reviewer
- Review coverage percentage

### Target Metrics
- **First review time**: < 24 hours (90th percentile)
- **Review cycles**: < 3 rounds average
- **Coverage**: 100% of code changes reviewed
- **Post-merge issues**: < 5% of reviewed PRs

## Special Cases

### Hotfixes
- Expedited review process (4-hour target)
- May have reduced review requirements
- Must include post-fix follow-up for full review
- Risk assessment required

### Documentation Changes
- Generally require single approval
- Technical accuracy verification needed
- Style and clarity feedback welcome
- Breaking changes need broader review

### Dependency Updates
- Security updates: Expedited review
- Major version bumps: Full compatibility review
- Minor updates: Automated testing sufficient
- New dependencies: Architecture team review

### Experimental Features
- May have relaxed review requirements
- Must be clearly marked as experimental
- Documentation must include stability warnings
- Feature flags should gate functionality

## Training and Resources

### New Team Members
- Shadow experienced reviewers for 2 weeks
- Practice reviews on closed PRs
- Receive feedback on review quality
- Gradual increase in review responsibilities

### Continuous Improvement
- Monthly review process retrospectives
- Annual training on security review practices
- Regular updates to guidelines
- Tool and automation improvements

### Resources
- [Effective Code Reviews (Google)](https://google.github.io/eng-practices/review/)
- [Security Code Review Guidelines](https://owasp.org/www-project-code-review-guide/)
- [Performance Review Best Practices](https://engineering.fb.com/2018/05/21/developer-tools/finding-and-fixing-software-bugs-automatically-with-sapfix-and-sapienz/)

## Contact and Feedback

### Questions
- **Process Questions**: engineering@async-toolformer.com
- **Tool Issues**: devops@async-toolformer.com  
- **Security Concerns**: security@async-toolformer.com

### Feedback
We welcome feedback on these guidelines:
- Create issue in the repository
- Discuss in team retrospectives
- Propose changes via PR to this document

*Last Updated: 2025-01-15*
*Next Review: 2025-04-15*