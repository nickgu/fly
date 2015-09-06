
VERSION = 1.1.0
COMPILER = g++
CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O3 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall
DEBUG_CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O0 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall

TARGET = fly
INCLUDES = -I./include \
		 -I./models
		  
LIBS = -lcrypto \
	   -lpthread

OBJECTS = fly_data.o

all: clean $(OBJECTS) $(TARGET)
	@echo 'MAKE: ALL'

$(OBJECTS): %.o:%.cc
	@echo -e "\033[40;32;1m --> make : [[" $@ " ]] <-- \033[0m"
	$(COMPILER) -c $^ -o $@ $(CPPFLAGS) $(INCLUDES)

fly: fly.cc $(OBJECTS)
	@echo -e "\033[40;32;1m --> make : [[" $@ " ]] <-- \033[0m"
	$(COMPILER) $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

fly_debug: fly.cc $(OBJECTS)
	@echo -e "\033[40;32;1m --> make : [[" $@ " ]] <-- \033[0m"
	$(COMPILER) $^ -o $@ $(LIBS) $(DEBUG_CPPFLAGS) $(INCLUDES) 

clean:
	rm -rf *.o *.so *~ $(TARGET) models/*~ include/*~ 

