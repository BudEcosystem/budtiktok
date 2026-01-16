# Release Process for BudTikTok Python Package

This document describes how to release new versions of the `budtiktok` Python package to PyPI.

## Prerequisites

### 1. PyPI Account Setup

1. Create accounts on:
   - **PyPI (Production)**: https://pypi.org/account/register/
   - **TestPyPI (Testing)**: https://test.pypi.org/account/register/

2. Generate API tokens:
   - Go to Account Settings → API tokens
   - Create a token for the `budtiktok` project (or account-wide token)
   - Save the token securely (you won't see it again)

### 2. GitHub Repository Secrets

Add the following secrets to your GitHub repository:

1. Go to: `Settings` → `Secrets and variables` → `Actions` → `New repository secret`

2. Add these secrets:
   - **`PYPI_API_TOKEN`**: Your PyPI API token (starts with `pypi-`)
   - **`TEST_PYPI_API_TOKEN`**: Your TestPyPI API token (for testing)

**How to add secrets:**
```
Repository Settings → Secrets and variables → Actions → New repository secret
Name: PYPI_API_TOKEN
Value: pypi-AgEIcHlwaS5vcmc... (your token)
```

## Release Workflow

### Option 1: Automated Release via GitHub Release (Recommended)

This is the recommended approach for production releases.

**Steps:**

1. **Update version number** in both files:
   ```bash
   # Edit version in pyproject.toml
   vim crates/budtiktok-python/pyproject.toml
   # Change: version = "0.1.0" → version = "0.2.0"

   # Edit version in Cargo.toml
   vim crates/budtiktok-python/Cargo.toml
   # Change: version = "0.1.0" → version = "0.2.0"
   ```

2. **Commit and push changes:**
   ```bash
   git add crates/budtiktok-python/pyproject.toml crates/budtiktok-python/Cargo.toml
   git commit -m "chore: bump version to 0.2.0"
   git push origin main
   ```

3. **Create a Git tag:**
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```

4. **Create GitHub Release:**
   - Go to: https://github.com/BudEcosystem/budtiktok/releases/new
   - Choose tag: `v0.2.0`
   - Release title: `v0.2.0`
   - Description: Add release notes (features, fixes, breaking changes)
   - Click: `Publish release`

5. **Monitor the workflow:**
   - Go to: `Actions` tab in GitHub
   - Watch "Build and Publish Python Package" workflow
   - It will:
     - Build wheels for Linux (x86_64, aarch64)
     - Build wheels for macOS (x86_64, aarch64)
     - Build wheels for Windows (x64)
     - Build source distribution (sdist)
     - Publish all to PyPI

6. **Verify on PyPI:**
   - Check: https://pypi.org/project/budtiktok/
   - Test installation: `pip install budtiktok==0.2.0`

### Option 2: Manual Testing via Workflow Dispatch

Use this to test wheel building without publishing.

**Steps:**

1. Go to: `Actions` → `Build and Publish Python Package` → `Run workflow`

2. Select branch: `main`

3. Choose publish option:
   - **`no`**: Build wheels and publish to TestPyPI (for testing)
   - **`yes`**: Build wheels and publish to PyPI (production)

4. Click: `Run workflow`

5. **Test from TestPyPI (if publish=no):**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ budtiktok
   ```

   Note: `--extra-index-url https://pypi.org/simple/` is needed for dependencies like numpy.

### Option 3: Local Testing and Building

For local development and testing before releasing.

**Prerequisites:**
```bash
pip install maturin
```

**Build locally:**
```bash
cd crates/budtiktok-python

# Development build (debug mode)
maturin develop

# Release build (optimized)
maturin develop --release

# Build wheel files
maturin build --release

# Build for multiple Python versions
maturin build --release --interpreter python3.8 python3.9 python3.10 python3.11 python3.12
```

**Test the wheel:**
```bash
# Install the built wheel
pip install target/wheels/budtiktok-0.2.0-*.whl

# Test it
python -c "from budtiktok import Tokenizer; print('Success!')"
```

**Publish manually to TestPyPI:**
```bash
maturin publish --repository testpypi
# Enter your TestPyPI credentials when prompted
```

**Publish manually to PyPI:**
```bash
maturin publish
# Enter your PyPI credentials when prompted
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Major version** (1.0.0): Breaking API changes
- **Minor version** (0.2.0): New features, backward compatible
- **Patch version** (0.1.1): Bug fixes, backward compatible

**Examples:**
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.0` → `0.2.0`: New feature added
- `0.9.0` → `1.0.0`: First stable release with API guarantees

## Release Checklist

Before creating a release:

- [ ] All tests pass (`cargo test`)
- [ ] Python tests pass (if any in `crates/budtiktok-python/tests/`)
- [ ] Version bumped in `pyproject.toml` and `Cargo.toml`
- [ ] CHANGELOG.md updated (if you maintain one)
- [ ] Documentation updated
- [ ] Benchmarks run and verified
- [ ] Manual smoke test performed

## Troubleshooting

### Build fails with "command not found: maturin"

```bash
pip install maturin
```

### Build fails with Rust compiler errors

```bash
# Update Rust
rustup update stable

# Clean build
cargo clean
```

### PyPI upload fails with "403 Forbidden"

- Check that `PYPI_API_TOKEN` secret is set correctly
- Verify the token has permissions for the `budtiktok` project
- Make sure the version doesn't already exist on PyPI

### Wheels don't work on certain platforms

- Check the `compatibility` setting in `pyproject.toml`
- For older Linux systems, use `manylinux2014` instead of `manylinux_2_28`
- For macOS universal binaries, use `--universal2` flag

### Import fails after installation

```bash
# Check Python version compatibility
python --version  # Must be >= 3.8

# Verify numpy is installed
pip install numpy>=1.20.0

# Check for architecture mismatches
python -c "import platform; print(platform.machine())"
```

## Platform-Specific Notes

### Linux (manylinux)

The workflow builds for:
- `x86_64` (Intel/AMD 64-bit)
- `aarch64` (ARM 64-bit, e.g., AWS Graviton)

Compatibility: `manylinux_2_28` (glibc 2.28+, CentOS 8+, Ubuntu 18.04+)

### macOS

The workflow builds for:
- `x86_64` (Intel Macs)
- `aarch64` (Apple Silicon M1/M2/M3)

Both architectures are built separately for maximum compatibility.

### Windows

The workflow builds for:
- `x64` (64-bit Windows)

## CI/CD Workflow Details

The `.github/workflows/python-release.yml` workflow:

1. **Triggers:**
   - Automatically on GitHub Release publication
   - Manually via workflow dispatch

2. **Jobs:**
   - `linux`: Builds wheels for Linux (x86_64, aarch64)
   - `macos`: Builds wheels for macOS (x86_64, aarch64)
   - `windows`: Builds wheels for Windows (x64)
   - `sdist`: Builds source distribution
   - `publish`: Publishes to PyPI (on release)
   - `publish-testpypi`: Publishes to TestPyPI (on manual trigger with publish=no)

3. **Optimizations:**
   - Uses `sccache` for faster Rust compilation
   - Builds in parallel across platforms
   - Uses official `PyO3/maturin-action` for consistent builds

## Support

For issues with releases:
- GitHub Issues: https://github.com/BudEcosystem/budtiktok/issues
- Check Actions logs for build failures
- Verify PyPI tokens and permissions
