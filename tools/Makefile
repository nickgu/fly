

VERSION = 1.1.0
COMPILER = g++
CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O3 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall
DEBUG_CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O0 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall

TARGET=auc test_gbdt binary_feature_less PyFly.so

OBJECTS= ../src/*.o

INCLUDES= -I../src/include\
		  -I../src/models

PYTHON27_INCLUDE= -I /home/users/gusimiu/.jumbo/include
		  
LIBS = -lcrypto \
	   -lpthread

all: clean $(TARGET)
	@echo 'MAKE: TOOLS'
	mkdir tools
	mv $(TARGET) tools

test_gbdt: test_gbdt.cc $(OBJECTS)
	@echo 'MAKE: TEST_GBDT'
	$(COMPILER) $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

binary_feature_less: binary_feature_less.cc $(OBJECTS)
	@echo 'MAKE: BINARY_FEATURE_LESS'
	$(COMPILER) $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

auc: auc.cc $(OBJECTS)
	@echo 'MAKE: AUC'
	$(COMPILER) $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

PyFly.so: pyfly.cc $(OBJECTS)
	$(COMPILER) -fpic $^ -shared -o $@ $(INCLUDES) $(PYTHON27_INCLUDE) $(LIBS) $(CPPFLAGS) 

clean:
	rm -rf *.o *.so *~ $(TARGET)
	rm -rf tools




