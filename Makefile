# Pendulum benchmark -- convenience wrapper around CMake.
#
#   make            configure and build everything that this machine can build
#   make bench      run every available implementation, report timings
#   make bench-quick    same, but skip the ones that take seconds per run
#   make validate   check that every implementation computes the same trajectory
#   make animate    regenerate pendulum.gif
#   make clean      remove the build directory
#
# Nothing here is required: the sources are ordinary single-file programs and
# scripts, and README.md shows the one-line command for each.

BUILD  ?= build
PYTHON ?= python3
CMAKE_ARGS ?=

# <mdspan> currently needs clang with libc++, so prefer clang++ when it exists.
# Override with `make CXX=g++` (the two mdspan variants are then skipped).
# Note: CXX always has a value in GNU Make, so ask where that value came from.
ifeq ($(origin CXX),default)
  CLANGXX := $(shell command -v clang++ 2>/dev/null)
  ifneq ($(CLANGXX),)
    CMAKE_ARGS += -DCMAKE_CXX_COMPILER=$(CLANGXX)
  endif
else
  CMAKE_ARGS += -DCMAKE_CXX_COMPILER=$(CXX)
endif

.PHONY: all build bench bench-quick validate animate clean help

all: build

build:
	@cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGS)
	@cmake --build $(BUILD) --parallel

bench: build
	@$(PYTHON) scripts/bench.py --build $(BUILD)

bench-quick: build
	@$(PYTHON) scripts/bench.py --build $(BUILD) --quick

validate: build
	@$(PYTHON) scripts/validate.py --build $(BUILD)

animate:
	@$(PYTHON) animate.py

clean:
	@rm -rf $(BUILD)

help:
	@sed -n '3,10p' $(MAKEFILE_LIST) | sed 's/^# \?//'
