# BudTikTok Multi-stage Dockerfile
# Task 1.2.6: Docker build pipeline
#
# Build targets:
#   - budtiktok-worker: Tokenization worker
#   - budtiktok-coordinator: Request coordinator
#   - budtiktok-gpu: GPU-enabled tokenization (requires CUDA)

# =============================================================================
# Stage 1: Build environment
# =============================================================================
FROM rust:1.75-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy workspace files for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/budtiktok-core/Cargo.toml ./crates/budtiktok-core/
COPY crates/budtiktok-simd/Cargo.toml ./crates/budtiktok-simd/
COPY crates/budtiktok-gpu/Cargo.toml ./crates/budtiktok-gpu/
COPY crates/budtiktok-ipc/Cargo.toml ./crates/budtiktok-ipc/
COPY crates/budtiktok-coordinator/Cargo.toml ./crates/budtiktok-coordinator/
COPY crates/budtiktok-cli/Cargo.toml ./crates/budtiktok-cli/
COPY crates/budtiktok-bench/Cargo.toml ./crates/budtiktok-bench/
COPY crates/budtiktok-accuracy/Cargo.toml ./crates/budtiktok-accuracy/

# Create dummy source files for dependency caching
RUN mkdir -p crates/budtiktok-core/src && echo "pub fn dummy() {}" > crates/budtiktok-core/src/lib.rs && \
    mkdir -p crates/budtiktok-simd/src && echo "pub fn dummy() {}" > crates/budtiktok-simd/src/lib.rs && \
    mkdir -p crates/budtiktok-gpu/src && echo "pub fn dummy() {}" > crates/budtiktok-gpu/src/lib.rs && \
    mkdir -p crates/budtiktok-ipc/src && echo "pub fn dummy() {}" > crates/budtiktok-ipc/src/lib.rs && \
    mkdir -p crates/budtiktok-coordinator/src && echo "pub fn dummy() {}" > crates/budtiktok-coordinator/src/lib.rs && \
    mkdir -p crates/budtiktok-cli/src && echo "fn main() {}" > crates/budtiktok-cli/src/main.rs && \
    mkdir -p crates/budtiktok-bench/src && echo "fn main() {}" > crates/budtiktok-bench/src/main.rs && \
    mkdir -p crates/budtiktok-accuracy/src && echo "fn main() {}" > crates/budtiktok-accuracy/src/main.rs

# Build dependencies only (cached layer)
RUN cargo build --release --workspace 2>/dev/null || true

# Copy actual source code
COPY . .

# Touch source files to invalidate cache
RUN find crates -name "*.rs" -exec touch {} \;

# Build release binaries
RUN cargo build --release --workspace

# =============================================================================
# Stage 2: Runtime base image
# =============================================================================
FROM debian:bookworm-slim AS runtime-base

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 budtiktok

WORKDIR /app

# =============================================================================
# Stage 3: Worker image
# =============================================================================
FROM runtime-base AS worker

COPY --from=builder /app/target/release/budtiktok-cli /usr/local/bin/budtiktok

USER budtiktok

ENV RUST_LOG=info
ENV BUDTIKTOK_MODE=worker

EXPOSE 8080

ENTRYPOINT ["budtiktok"]
CMD ["serve", "--port", "8080"]

# =============================================================================
# Stage 4: Coordinator image
# =============================================================================
FROM runtime-base AS coordinator

COPY --from=builder /app/target/release/budtiktok-cli /usr/local/bin/budtiktok

USER budtiktok

ENV RUST_LOG=info
ENV BUDTIKTOK_MODE=coordinator

EXPOSE 8080
EXPOSE 9090

ENTRYPOINT ["budtiktok"]
CMD ["coordinator", "--port", "8080", "--metrics-port", "9090"]

# =============================================================================
# Stage 5: GPU-enabled worker (CUDA)
# =============================================================================
FROM nvidia/cuda:12.3-runtime-ubuntu22.04 AS gpu-base

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 budtiktok

WORKDIR /app

FROM gpu-base AS gpu-worker

# Copy from builder (note: requires GPU-enabled build)
COPY --from=builder /app/target/release/budtiktok-cli /usr/local/bin/budtiktok

USER budtiktok

ENV RUST_LOG=info
ENV BUDTIKTOK_MODE=worker
ENV BUDTIKTOK_GPU=auto
ENV CUDA_VISIBLE_DEVICES=all

EXPOSE 8080

ENTRYPOINT ["budtiktok"]
CMD ["serve", "--port", "8080", "--gpu"]

# =============================================================================
# Default target: worker
# =============================================================================
FROM worker AS default
