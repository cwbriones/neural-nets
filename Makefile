PROJECT=neuralnets

all:
	mkdir -p ./bin
	g++ -I/usr/include/eigen3 src/* -o bin/${PROJECT} -std=c++11
