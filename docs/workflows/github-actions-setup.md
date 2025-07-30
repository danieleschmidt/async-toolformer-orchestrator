# GitHub Actions Workflows Setup

This document provides templates and setup instructions for implementing GitHub Actions workflows. These workflows must be manually created in `.github/workflows/` directory due to security restrictions.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Run tests, linting, and type checking on every PR and push.

**Template Location**: Create as `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run linting
      run: make lint
    
    - name: Run type checking
      run: make typecheck
    
    - name: Run unit tests
      run: make test-unit
    
    - name: Run integration tests
      run: make test-integration
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. Security Scanning (`security.yml`)

**Purpose**: Automated security vulnerability scanning and SBOM generation.

```yaml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        pip install safety pip-audit
    
    - name: Run Bandit security scan
      run: bandit -r src/ -f json -o bandit-report.json || true
    
    - name: Run Safety check
      run: safety check --json --output safety-report.json || true
    
    - name: Run pip-audit
      run: pip-audit --format=json --output=audit-report.json || true
    
    - name: Generate SBOM
      run: python scripts/sbom/generate-sbom.py
    
    - name: Upload security artifacts
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          audit-report.json
          sbom.json
```

### 3. Release Automation (`release.yml`)

**Purpose**: Automated releases to PyPI with proper versioning and changelogs.

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install build dependencies
      run: |
        pip install build twine
    
    - name: Build package
      run: make build
    
    - name: Verify package
      run: twine check dist/*
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
```

### 4. Container Build (`docker.yml`)

**Purpose**: Build and publish Docker images with security scanning.

```yaml
name: Docker Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

jobs:
  docker:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Docker Hub
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: yourusername/async-toolformer-orchestrator
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ steps.meta.outputs.tags }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

## Workflow Implementation Checklist

### Prerequisites
- [ ] Create `.github/workflows/` directory
- [ ] Add repository secrets for PyPI and Docker Hub
- [ ] Configure branch protection rules
- [ ] Set up Codecov integration (optional)

### Required Secrets
Add these secrets in repository settings:

```bash
PYPI_API_TOKEN          # PyPI publishing token
DOCKER_USERNAME         # Docker Hub username  
DOCKER_PASSWORD         # Docker Hub token
CODECOV_TOKEN          # Codecov upload token (optional)
```

### Branch Protection Rules
Recommended settings for `main` branch:
- Require PR reviews (2 reviewers)
- Require status checks to pass
- Require branches to be up to date
- Include administrators in restrictions
- Allow force pushes: No
- Allow deletions: No

## Advanced Workflow Features

### Matrix Testing
The CI workflow includes Python version matrix (3.10, 3.11, 3.12).

### Caching Strategy
- Pip cache for faster dependency installation
- Docker layer caching for image builds

### Security Integration
- Automated dependency vulnerability scanning
- Container image security scanning
- SBOM generation for compliance

### Release Management
- Automated PyPI publishing on tagged releases
- GitHub release notes generation
- Docker image tagging aligned with Git tags

## Monitoring and Observability

### Workflow Monitoring
Monitor workflow success rates and duration:
- Set up alerts for failed workflows
- Track build time trends
- Monitor dependency update frequency

### Quality Gates
Workflows enforce quality standards:
- 85% minimum code coverage
- Zero high-severity security vulnerabilities
- All linting and type checking must pass

## Troubleshooting

### Common Issues
1. **Permission denied on PyPI**: Verify API token scopes
2. **Docker push fails**: Check credentials and repository access
3. **Tests timeout**: Adjust timeout values in workflow files
4. **Cache misses**: Verify cache key patterns match dependencies

### Debugging Tips
- Enable workflow debug logging with `ACTIONS_STEP_DEBUG` secret
- Use `act` tool for local workflow testing
- Check workflow run logs for detailed error information

## Security Considerations

### Secrets Management
- Use repository secrets, not hardcoded values
- Rotate secrets regularly
- Limit secret access to necessary workflows only

### Dependency Security
- Pin action versions to specific commits or tags
- Regularly update action versions
- Review third-party action permissions

### Container Security
- Use minimal base images
- Scan images for vulnerabilities
- Follow Docker security best practices

## Performance Optimization

### Build Speed
- Use build caches effectively
- Parallelize independent jobs
- Optimize Docker layer ordering

### Resource Usage
- Monitor workflow resource consumption
- Use appropriate runner sizes
- Consider self-hosted runners for heavy workloads

## Compliance and Audit

### Audit Trail
All workflows provide comprehensive audit trails:
- Git commit information
- Build artifacts with checksums
- Security scan results
- Deployment records

### Compliance Features
- SLSA Level 2+ compliance ready
- SBOM generation for supply chain security
- Reproducible builds with pinned dependencies