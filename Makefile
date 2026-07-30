.PHONY: build server test test-metal test-model smoke bench-chunk bench-concurrency bench-engines bench-prefix profile mutate bench lint fmt venv model model-gguf fixtures reference

MODEL     ?= $(CURDIR)/models/Qwen3-0.6B
REFERENCE ?= $(CURDIR)/models/reference
PYTHON    ?= $(CURDIR)/.venv/bin/python
HF        ?= https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main
GGUF      ?= https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf

build:
	cargo build --release --features metal

server: build
	./target/release/pagedllm-server --model $(MODEL)

# The CPU path, which is what CI runs and what every kernel is checked against.
test:
	cargo test

# Adds the tests that need a Metal device. Local only: GitHub's macOS runners
# have no GPU to dispatch to.
test-metal:
	cargo test --features metal

# The full-scale comparison against the reference implementation. Needs the
# checkpoint and both reference dumps, which are not in the repository; `make
# model reference` produces them.
test-model:
	PAGEDLLM_MODEL_DIR=$(MODEL) \
	PAGEDLLM_REFERENCE_DIR=$(REFERENCE) \
	PAGEDLLM_REFERENCE_BF16_DIR=$(REFERENCE)-bf16 \
	cargo test --release --features metal --test reference_model -- --nocapture

# Drives the server over HTTP, on a process it starts and stops itself, and
# prints the throughput of one sequence against the contiguous cache.
smoke: build
	python3 scripts/smoke-server.py

# This engine against the others on this machine, driven by guidellm.
bench-engines: build
	$(PYTHON) scripts/bench-engines.py

# What a shared prompt buys, and what it costs when nothing is shared.
bench-prefix: build
	$(PYTHON) scripts/bench-prefix.py

# A Metal System Trace of a decode under load, written to docs/. Needs a full
# Xcode, which nothing else here does.
profile: build
	$(PYTHON) scripts/profile-gpu.py

# What continuous batching buys against what the reservation costs.
bench-concurrency: build
	python3 scripts/bench-concurrency.py

## What a long prompt does to the sequences already generating, with the pass
## budget on and off.
bench-chunk: build
	python3 scripts/bench-chunk.py

# Puts each defect the forward-pass tests exist for back, and checks they fail.
# Adds the full-scale suite when the checkpoint and its reference dumps are
# there; without them the committed fixture carries it alone.
mutate:
	@if [ -d "$(MODEL)" ] && [ -d "$(REFERENCE)" ] && [ -d "$(REFERENCE)-bf16" ]; then \
		PAGEDLLM_MODEL_DIR=$(MODEL) \
		PAGEDLLM_REFERENCE_DIR=$(REFERENCE) \
		PAGEDLLM_REFERENCE_BF16_DIR=$(REFERENCE)-bf16 \
		python3 scripts/mutate.py; \
	else \
		python3 scripts/mutate.py; \
	fi

bench:
	cargo bench

lint:
	cargo fmt --all --check
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features metal -- -D warnings

fmt:
	cargo fmt --all
	cargo clippy --all-targets --fix --allow-dirty

# Everything below produces what the tests read and is not committed: torch and
# transformers for the oracle, 1.5 GB of weights, and the dumps taken from them.
venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install --quiet --upgrade pip
	$(PYTHON) -m pip install --quiet torch transformers guidellm

model:
	mkdir -p $(MODEL)
	cd $(MODEL) && for f in config.json model.safetensors tokenizer.json tokenizer_config.json generation_config.json; do \
		curl -fsSL -o $$f $(HF)/$$f; \
	done

# Regenerates the fixture that is committed, so a change to the oracle is a
# change to the repository rather than to one machine.
# The same weights in the format llama.cpp reads, for the comparison to be of
# one model rather than of two.
model-gguf:
	mkdir -p $(CURDIR)/models/gguf
	curl -fsSL -o $(CURDIR)/models/gguf/Qwen3-0.6B-BF16.gguf $(GGUF)

fixtures:
	$(PYTHON) scripts/dump_reference.py tiny crates/pagedllm/tests/fixtures/tiny

reference:
	$(PYTHON) scripts/dump_reference.py real $(MODEL) $(REFERENCE) float32
	$(PYTHON) scripts/dump_reference.py real $(MODEL) $(REFERENCE)-bf16 bfloat16
