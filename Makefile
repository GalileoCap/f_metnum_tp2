IPATH := src/tp2lib.cpp
OPATH := analysis/tp2$(shell python3-config --extension-suffix)
CFLAGS := -O3 -Wall -shared -std=c++17 -fPIC -fopenmp $(shell python3-config --includes) -I./extern/pybind11/include/ -I./extern/eigen/

all:
	g++ $(CFLAGS) $(IPATH) -o $(OPATH)
clean:
	rm $(OPATH) 
