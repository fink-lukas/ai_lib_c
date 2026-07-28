# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -O3 -march=native
LDFLAGS = -lm

# Directories
SRC_DIR = src
BUILD_DIR = build

# Find all .c files in src/
SRCS = $(wildcard $(SRC_DIR)/*.c)
TARGET = $(BUILD_DIR)/program

# Default rule
all: $(TARGET)

# Rule to link the program
$(TARGET): $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Clean rule
clean:
	rm -rf $(BUILD_DIR)