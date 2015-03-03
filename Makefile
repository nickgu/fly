
VERSION = 1.1.0
CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O3 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall
DEBUG_CPPFLAGS =  -D__VERSION_ID__="\"$(VERSION)\"" -g -Wall -O0 -fPIC  -pipe -D_REENTRANT -DLINUX -Wall

TARGET=fly auc fly_debug test_gbdt
INCLUDES=
		  
LIBS = -lcrypto \
	   -lpthread

all: clean $(TARGET)
	@echo 'MAKE: ALL'
	mkdir output
	mv $(TARGET) output

fly: fly.cc
	@echo 'MAKE: FLY'
	g++ $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

fly_debug: fly.cc
	@echo 'MAKE: FLY_DEBUG'
	g++ $^ -o $@ $(LIBS) $(DEBUG_CPPFLAGS) $(INCLUDES) 

test_gbdt: test_gbdt.cc
	@echo 'MAKE: TEST_GBDT'
	g++ $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

auc: auc.cc
	@echo 'MAKE: AUC'
	g++ $^ -o $@ $(LIBS) $(CPPFLAGS) $(INCLUDES) 

clean:
	rm -rf *.o *.so *~ $(TARGET)
	rm -rf output

test:
	make c_prod.so
	python test_prod.py < item.txt


