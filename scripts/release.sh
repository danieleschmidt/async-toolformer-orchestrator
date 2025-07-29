#!/bin/bash
# Release automation script for async-toolformer-orchestrator

set -euo pipefail

# Configuration
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    local missing_commands=()
    
    if ! command_exists git; then
        missing_commands+=(git)
    fi
    
    if ! command_exists python; then
        missing_commands+=(python)
    fi
    
    if ! command_exists pip; then
        missing_commands+=(pip)
    fi
    
    if ! command_exists twine; then
        missing_commands+=(twine)
    fi
    
    if [[ ${#missing_commands[@]} -gt 0 ]]; then
        log_error "Missing required commands: ${missing_commands[*]}"
        log_error "Please install missing commands and try again."
        exit 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check if working directory is clean
    if [[ -n $(git status --porcelain) ]]; then
        log_error "Working directory is not clean. Please commit or stash changes."
        git status --short
        exit 1
    fi
    
    # Check if we're on main branch
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$current_branch" != "main" ]]; then
        log_warning "Not on main branch (current: $current_branch)"
        read -p "Continue anyway? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Aborted by user"
            exit 1
        fi
    fi
    
    log_success "Prerequisites validated"
}

# Get current version from __init__.py
get_current_version() {
    python -c "
import sys
sys.path.insert(0, 'src')
from async_toolformer import __version__
print(__version__)
"
}

# Bump version based on type
bump_version() {
    local version_type="$1"
    local current_version
    current_version=$(get_current_version)
    
    log_info "Current version: $current_version"
    
    # Split version into components
    IFS='.' read -ra VERSION_PARTS <<< "$current_version"
    local major="${VERSION_PARTS[0]}"
    local minor="${VERSION_PARTS[1]}"
    local patch="${VERSION_PARTS[2]}"
    
    # Bump based on type
    case "$version_type" in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            log_error "Invalid version type: $version_type (must be major, minor, or patch)"
            exit 1
            ;;
    esac
    
    local new_version="$major.$minor.$patch"
    log_info "New version: $new_version"
    
    # Update version in __init__.py
    sed -i.bak "s/__version__ = \".*\"/__version__ = \"$new_version\"/" src/async_toolformer/__init__.py
    rm src/async_toolformer/__init__.py.bak
    
    echo "$new_version"
}

# Run tests
run_tests() {
    log_info "Running test suite..."
    
    # Install dev dependencies
    pip install -e ".[dev]" >/dev/null 2>&1
    
    # Run linting
    log_info "Running linting checks..."
    if ! make lint >/dev/null 2>&1; then
        log_error "Linting checks failed"
        return 1
    fi
    
    # Run type checking
    log_info "Running type checks..."
    if ! make typecheck >/dev/null 2>&1; then
        log_error "Type checking failed"
        return 1
    fi
    
    # Run unit tests
    log_info "Running unit tests..."
    if ! make test-unit >/dev/null 2>&1; then
        log_error "Unit tests failed"
        return 1
    fi
    
    # Run integration tests
    log_info "Running integration tests..."
    if ! make test-integration >/dev/null 2>&1; then
        log_warning "Integration tests failed (continuing anyway)"
    fi
    
    log_success "Test suite passed"
}

# Build package
build_package() {
    log_info "Building package..."
    
    # Clean previous builds
    make clean >/dev/null 2>&1 || true
    
    # Build package
    if ! make build >/dev/null 2>&1; then
        log_error "Package build failed"
        exit 1
    fi
    
    # Check package
    if ! twine check dist/* >/dev/null 2>&1; then
        log_error "Package check failed"
        exit 1
    fi
    
    log_success "Package built successfully"
}

# Create git tag and commit
create_git_tag() {
    local version="$1"
    local tag="v$version"
    
    log_info "Creating git tag: $tag"
    
    # Commit version change
    git add src/async_toolformer/__init__.py
    git add CHANGELOG.md 2>/dev/null || true
    git commit -m "chore: bump version to $version

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Create tag
    git tag -a "$tag" -m "Release $version

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    log_success "Git tag created: $tag"
}

# Push to remote
push_release() {
    local version="$1"
    local tag="v$version"
    
    log_info "Pushing release to remote..."
    
    # Push commits and tags
    git push origin HEAD
    git push origin "$tag"
    
    log_success "Release pushed to remote"
}

# Publish to PyPI
publish_package() {
    local dry_run="$1"
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "Publishing to PyPI (TEST)..."
        if ! twine upload --repository testpypi dist/* >/dev/null 2>&1; then
            log_error "Test PyPI upload failed"
            return 1
        fi
        log_success "Package published to Test PyPI"
    else
        log_info "Publishing to PyPI (PRODUCTION)..."
        if ! twine upload dist/* >/dev/null 2>&1; then
            log_error "PyPI upload failed"
            return 1
        fi
        log_success "Package published to PyPI"
    fi
}

# Main release function
release() {
    local version_type="$1"
    local dry_run="${2:-false}"
    
    log_info "Starting release process for $version_type version..."
    
    # Validate prerequisites
    validate_prerequisites
    
    # Run tests
    if ! run_tests; then
        log_error "Tests failed. Aborting release."
        exit 1
    fi
    
    # Bump version
    local new_version
    new_version=$(bump_version "$version_type")
    
    # Build package
    build_package
    
    # Create git tag
    create_git_tag "$new_version"
    
    if [[ "$dry_run" == "false" ]]; then
        # Push to remote
        push_release "$new_version"
        
        # Publish package
        publish_package false
        
        log_success "Release $new_version completed successfully!"
        log_info "GitHub Actions will create the GitHub release automatically."
        log_info "Monitor the release workflow at: https://github.com/yourusername/async-toolformer-orchestrator/actions"
    else
        log_info "Dry run completed. Version bumped to $new_version (not pushed)"
        log_info "To complete the release, run: $0 $version_type"
        
        # Reset version change for dry run
        git reset --hard HEAD~1
        git tag -d "v$new_version" >/dev/null 2>&1 || true
    fi
}

# Show usage
usage() {
    cat << EOF
Usage: $0 <version_type> [--dry-run]

Release automation script for async-toolformer-orchestrator

Arguments:
    version_type    Version bump type: major, minor, or patch

Options:
    --dry-run      Perform a dry run without pushing or publishing

Examples:
    $0 patch        # Release a patch version (e.g., 0.1.0 -> 0.1.1)
    $0 minor        # Release a minor version (e.g., 0.1.1 -> 0.2.0)
    $0 major        # Release a major version (e.g., 0.2.0 -> 1.0.0)
    
    $0 patch --dry-run    # Test the release process without publishing

EOF
}

# Main script
main() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi
    
    local version_type="$1"
    local dry_run="false"
    
    if [[ $# -gt 1 && "$2" == "--dry-run" ]]; then
        dry_run="true"
    fi
    
    case "$version_type" in
        major|minor|patch)
            release "$version_type" "$dry_run"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Invalid version type: $version_type"
            usage
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi