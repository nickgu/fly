
VERSION = 1.1.0
COMPILER = g++
CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O3 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall
DEBUG_CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O0 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall

TARGET=fly
INCLUDES=-I./include \
		 -I./models
		  
LIBS = -lcrypto \
	   -lpthread

all: clean $(TARGET)
	@echo 'MAKE: ALL'

fly: fly.cc
	@echo 'MAKE: FLY'
	$(COMPILER) $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

fly_debug: fly.cc
	@echo 'MAKE: FLY_DEBUG'
	$(COMPILER) $^ -o $@ $(LIBS) $(DEBUG_CPPFLAGS) $(INCLUDES) 

clean:
	rm -rf *.o *.so *~ $(TARGET) models/*~ include/*~ 

