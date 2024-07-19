.PHONY = clean all

SOURCES := $(wildcard *.cu)
OUTPUTS := $(SOURCES:.cu=)

all : $(OUTPUTS)

$(OUTPUTS) : % : %.cu
	nvcc $^ --output-file $@

test :
	@echo ciao!
	touch t.j

clean :
	rm --force $(OUTPUTS)
