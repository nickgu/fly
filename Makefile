

VERSION = 1.1.0
COMPILER = g++
CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O3 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall
DEBUG_CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O0 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall

TARGET=src/fly tools/tools
		  
all: clean $(TARGET)
	@echo 'MAKE: ALL'
	mkdir output
	mv src/fly output/
	mv tools/tools output/
	cp README.md output
	cp -r conf output/

tools/tools:
	cd tools && make;
	cd ../

src/fly:
	cd src && make;
	cd ../

clean:
	rm -rf output
	cd src && make clean;
	cd ../;
	cd tools && make clean;

