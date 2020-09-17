CC=g++
SRC := $(ML_STARK_ROOT)/KNN/src
CFLAGS := -shared -std=c++11 -DEUCLID 
INCLUDE_DIR := $(PWD)/include/

all : main

main : $(SRC)/KNN.cc
	$(CC) $(CFLAGS) $(SRC)/KNN.cc -o main -L$(ML_STARK_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(ML_STARK_ROOT)/include -ldata

clean :
	rm main
