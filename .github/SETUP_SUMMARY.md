# BudTikTok Python Package - Setup Complete ✅

This document summarizes the pip installation setup for the BudTikTok project.

## What Was Created

### 1. GitHub Actions Workflows

**Location:** `.github/workflows/`

#### `python-release.yml` - Production Release Pipeline
- **Triggers:**
  - Automatically on GitHub Release publication
  - Manually via workflow dispatch
- **Builds:**
  - Linux wheels (x86_64, aarch64)
  - macOS wheels (x86_64 Intel, aarch64 Apple Silicon)
  - Windows wheels (x64)
  - Source distribution (sdist)
- **Publishes:**
  - To PyPI on release
  - To TestPyPI on manual trigger (for testing)

#### `python-test.yml` - CI Testing Pipeline
- **Triggers:**
  - On pull requests affecting Python code
  - On pushes to main branch
- **Tests:**
  - Python 3.8, 3.9, 3.10, 3.11, 3.12
  - Ubuntu, macOS, Windows
  - Import tests, basic functionality tests
  - pytest (if test files exist)
- **Builds:**
  - Verifies wheel builds on all platforms
  - Uploads test artifacts

### 2. Documentation

#### `.github/RELEASE.md` - Complete Release Guide
Comprehensive documentation covering:
- PyPI account setup
- GitHub secrets configuration
- Three release methods (automated, manual, local)
- Version numbering guidelines
- Release checklist
- Troubleshooting guide
- Platform-specific notes

#### Updated `crates/budtiktok-python/README.md`
Added sections:
- Development setup
- Building from source
- Multiple installation options
- Platform support matrix
- Testing instructions
- Contribution guidelines

## Your Current Package Configuration

**Package Name:** `budtiktok`
**Current Version:** `0.1.0`
**Python Support:** 3.8+
**Dependencies:** numpy>=1.20.0
**Optional:** torch>=1.9.0

**Build System:** Maturin (PyO3)
**Compatibility:** manylinux_2_28 (Linux), universal (macOS/Windows)

## Next Steps to Enable `pip install budtiktok`

### Step 1: Create PyPI Accounts (Required)

1. **Production PyPI:**
   - Register at: https://pypi.org/account/register/
   - Verify your email

2. **Test PyPI (Optional but recommended):**
   - Register at: https://test.pypi.org/account/register/
   - Verify your email

### Step 2: Generate API Tokens

1. **PyPI Production Token:**
   - Login to https://pypi.org
   - Go to: Account Settings → API tokens
   - Click: "Add API token"
   - Token name: `budtiktok-github-actions`
   - Scope: "Entire account" (or create project first for project-scoped token)
   - Copy the token (starts with `pypi-`)

2. **TestPyPI Token (Optional):**
   - Login to https://test.pypi.org
   - Repeat same process
   - Copy the token (starts with `pypi-`)

### Step 3: Add GitHub Secrets

1. Go to your GitHub repository
2. Navigate to: `Settings` → `Secrets and variables` → `Actions`
3. Click: `New repository secret`
4. Add secrets:

   **Secret 1:**
   - Name: `PYPI_API_TOKEN`
   - Value: `pypi-AgEIcHlwaS5vcmc...` (your PyPI token)

   **Secret 2 (Optional):**
   - Name: `TEST_PYPI_API_TOKEN`
   - Value: `pypi-AgEIcHlwaS5vcmc...` (your TestPyPI token)

### Step 4: Test the Setup (Optional but Recommended)

Before making a production release, test the workflow:

```bash
# 1. Trigger manual workflow dispatch
# Go to: Actions → "Build and Publish Python Package" → "Run workflow"
# Select: publish = "no" (publishes to TestPyPI)

# 2. Wait for workflow to complete

# 3. Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            budtiktok

# 4. Verify it works
python -c "from budtiktok import Tokenizer; print('Success!')"
```

### Step 5: Make Your First Release

**Option A: Via GitHub Release (Recommended)**

```bash
# 1. Update version
vim crates/budtiktok-python/pyproject.toml  # version = "0.1.0"
vim crates/budtiktok-python/Cargo.toml      # version = "0.1.0"

# 2. Commit and tag
git add crates/budtiktok-python/*.toml
git commit -m "chore: prepare v0.1.0 release"
git push origin main

git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

# 3. Create GitHub Release
# Go to: https://github.com/BudEcosystem/budtiktok/releases/new
# Tag: v0.1.0
# Title: v0.1.0
# Description: Add release notes
# Click: "Publish release"

# 4. Workflow runs automatically
# Monitor at: Actions tab

# 5. Verify on PyPI
# Check: https://pypi.org/project/budtiktok/
# Test: pip install budtiktok
```

**Option B: Manual Local Publishing**

```bash
cd crates/budtiktok-python

# Build wheels
maturin build --release

# Publish
maturin publish
# Enter PyPI username/token when prompted
```

## After First Release

Once published, users can install with:

```bash
pip install budtiktok
```

## Automated CI/CD

After setup, your release process is automated:

1. **On Pull Request:**
   - Builds and tests package
   - Tests on Python 3.8-3.12
   - Tests on Linux, macOS, Windows
   - Prevents merging broken code

2. **On Release:**
   - Automatically builds wheels for all platforms
   - Publishes to PyPI
   - No manual intervention needed

## Files Created/Modified

```
.github/
├── workflows/
│   ├── python-release.yml    # NEW: Release automation
│   └── python-test.yml        # NEW: CI testing
├── RELEASE.md                 # NEW: Release documentation
└── SETUP_SUMMARY.md          # NEW: This file

crates/budtiktok-python/
└── README.md                  # UPDATED: Added development section
```

## Quick Reference Commands

```bash
# Local development
cd crates/budtiktok-python
maturin develop --release

# Build wheel locally
maturin build --release

# Run tests
pytest crates/budtiktok-python/tests -v

# Publish to TestPyPI (manual)
maturin publish --repository testpypi

# Publish to PyPI (manual)
maturin publish

# Test installation from PyPI
pip install budtiktok

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            budtiktok
```

## Support & Troubleshooting

- **Detailed guide:** See `.github/RELEASE.md`
- **GitHub Issues:** https://github.com/BudEcosystem/budtiktok/issues
- **Maturin docs:** https://www.maturin.rs/

## Security Note

- **NEVER commit API tokens** to the repository
- Store tokens only in GitHub Secrets
- Use project-scoped tokens when possible
- Rotate tokens periodically

---

## Summary

✅ GitHub Actions workflows configured
✅ Documentation created
✅ Package already configured (`pyproject.toml` exists)
🔜 Setup PyPI accounts and tokens
🔜 Make your first release

Your project is now ready for pip distribution!
