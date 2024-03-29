GXX = g++
GFLAGS = -std=c++17
TARGET = bin/ocr
HEADERS = Utils.cpp
H_DIR = ./include/
DIR = ./src/

.PHONY: clean build pack

build:clean
	$(GXX) $(GFLAGS) $(DIR)main.cpp $(H_DIR)$(HEADERS) -o $(TARGET)
	
debug:
	$(GXX) $(GFLAGS) -g $(DIR)main.cpp $(H_DIR)$(HEADERS) -o $(TARGET)
	
clean:
	rm -rvf $(TARGET); clear

run: build
	$(TARGET)
