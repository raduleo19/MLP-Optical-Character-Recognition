GXX = g++
GFLAGS = -std=c++14 -O3
TARGET = ./bin/ocr
HEADERS = Utils.cpp
H_DIR = ./include/
DIR = ./src/

.PHONY: clean build pack

build:
	$(GXX) $(GFLAGS) $(DIR)main.cpp $(H_DIR)$(HEADERS) -o $(TARGET)
	
debug:
	$(GXX) $(GFLAGS) $(DIR)main.cpp $(H_DIR)$(HEADERS) -o -g $(TARGET)
	
clean:
	rm -rvf $(TARGET)

run: $(TARGET)
	$(TARGET)