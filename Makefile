MODULE_TOPDIR = $(shell grass --config path)

PGM = r.in.snodas

include $(MODULE_TOPDIR)/include/Make/Script.make

default: script
