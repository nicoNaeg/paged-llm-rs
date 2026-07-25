.PHONY: build server test test-metal bench lint fmt

build:
	cargo build --release --features metal

server: build
	./target/release/pagedllm-server

# The CPU path, which is what CI runs and what every kernel is checked against.
test:
	cargo test

# Adds the tests that need a Metal device. Local only: GitHub's macOS runners
# have no GPU to dispatch to.
test-metal:
	cargo test --features metal

bench:
	cargo bench

lint:
	cargo fmt --all --check
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features metal -- -D warnings

fmt:
	cargo fmt --all
	cargo clippy --all-targets --fix --allow-dirty
