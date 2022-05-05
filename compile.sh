#!/bin/sh

tar -xvf ./src/eigen.tar.gz; mv ./eigen-3.4.0 ./src/eigen
g++ -I ./src/eigen ./src/main.cc -o tp2
