PROJECT=neuralnets

all: exec_dir
	mkdir -p ./bin
	g++ -I/usr/include/eigen3 -std=c++11 src/* -o bin/${PROJECT}

osx: exec_dir
	g++-4.8 -I/usr/local/Cellar/eigen/3.2.1/include/eigen3/ src/* -std=c++11 -o bin/${PROJECT}

exec_dir:
	mkdir -p ./bin
