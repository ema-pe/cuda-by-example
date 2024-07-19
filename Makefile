.PHONY = all clean

all :
	$(MAKE) -C 03 all
	$(MAKE) -C 04 all
	$(MAKE) -C 05 all
	$(MAKE) -C 06 all
	$(MAKE) -C 07 all
	$(MAKE) -C 09 all
	$(MAKE) -C 10 all
	$(MAKE) -C 11 all
	$(MAKE) -C A all

clean :
	$(MAKE) -C 03 clean
	$(MAKE) -C 04 clean
	$(MAKE) -C 05 clean
	$(MAKE) -C 06 clean
	$(MAKE) -C 07 clean
	$(MAKE) -C 09 clean
	$(MAKE) -C 10 clean
	$(MAKE) -C 11 clean
	$(MAKE) -C A clean
