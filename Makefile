DIRS := apps
.PHONY: $(DIRS)

all: $(DIRS:%=subdir-all-%)

clean: $(DIRS:%=subdir-clean-%)

subdir-all-%:
	$(MAKE) -C $* all

subdir-clean-%:
	$(MAKE) -C $* clean
