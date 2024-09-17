help:
	@echo "make all -- make all in ${DIRS}"
	@echo "make clean -- make clean in ${DIRS}"
	@echo "make serve-docs -- serve the sycamore docs at http://localhost:8000/"

DIRS := apps
.PHONY: $(DIRS)

all: $(DIRS:%=subdir-all-%)

clean: $(DIRS:%=subdir-clean-%)

subdir-all-%:
	$(MAKE) -C $* all

subdir-clean-%:
	$(MAKE) -C $* clean

serve-docs:
	(cd docs && make serve-docs)
