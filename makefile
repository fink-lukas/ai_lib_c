# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -lm

# Directories
SRC_DIR = src
BUILD_DIR = build

# Find all .c files in src/
SRCS = $(wildcard $(SRC_DIR)/*.c)
# Define the final executable name inside the build directory
TARGET = $(BUILD_DIR)/program

# Default rule
all: $(TARGET)

# Rule to link the program
$(TARGET): $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

# Clean rule to remove the build directory
clean:
	rm -rf $(BUILD_DIR)