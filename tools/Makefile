

VERSION = 1.1.0
COMPILER = g++
CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O3 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall
DEBUG_CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O0 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall

TARGET=auc test_gbdt binary_feature_less
INCLUDES= -I../src/include\
		  -I../src/models
		  
LIBS = -lcrypto \
	   -lpthread

all: clean $(TARGET)
	@echo 'MAKE: TOOLS'
	mkdir tools
	mv $(TARGET) tools

fly: fly.cc
	@echo 'MAKE: FLY'
	$(COMPILER) $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

fly_debug: fly.cc
	@echo 'MAKE: FLY_DEBUG'
	$(COMPILER) $^ -o $@ $(LIBS) $(DEBUG_CPPFLAGS) $(INCLUDES) 

test_gbdt: test_gbdt.cc
	@echo 'MAKE: TEST_GBDT'
	$(COMPILER) $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

binary_feature_less: binary_feature_less.cc
	@echo 'MAKE: BINARY_FEATURE_LESS'
	$(COMPILER) $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

auc: auc.cc
	@echo 'MAKE: AUC'
	$(COMPILER) $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

clean:
	rm -rf *.o *.so *~ $(TARGET)
	rm -rf tools




