#!/bin/sh

unzip ./data/digit-recognizer.zip -d ./data
tar -xvf ./src/eigen.tar.gz; mv ./eigen-3.4.0 ./src/eigen
g++ -I ./src/eigen ./src/main.cc -o tp2
