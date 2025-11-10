# ===============================
# Compiler and target
# ===============================
CXX := clang++
TARGET := CS5990

# ===============================
# Paths
# ===============================
SRC_DIR := src
APP_DIR := app
INCLUDE_DIR := $(CURDIR)/include
BUILD_DIR := build

# ===============================
# Source / object files
# ===============================
SRCS := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(APP_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
OBJS := $(patsubst $(APP_DIR)/%.cpp, $(BUILD_DIR)/app_%.o, $(OBJS))

# ===============================
# Profiler Option
# Usage: make PROFILING=1
# This sets the compile-time macro PROFILER for your header guard.
# ===============================
ifeq ($(PROFILING),1)
  PROFILING_FLAGS := -DPROFILER
else
  PROFILING_FLAGS :=
  # Remove ChronoProfiler.cpp if profiling is disabled
  SRCS := $(filter-out $(SRC_DIR)/ChronoProfiler.cpp, $(SRCS))
  OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
  OBJS := $(patsubst $(APP_DIR)/%.cpp, $(BUILD_DIR)/app_%.o, $(OBJS))
endif

# ===============================
# Detect OS
# ===============================
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  VULKAN_INC := $(VULKAN_SDK)/include
  VULKAN_LIB := $(VULKAN_SDK)/lib
  GLM_INC := /opt/homebrew/include
else ifeq ($(UNAME_S),Linux)
  VULKAN_INC := /usr/include
  VULKAN_LIB := /usr/lib
  GLM_INC := /usr/include
endif

# Local include for stb
STB_INC := $(HOME)/include

# ===============================
# Compiler flags
# ===============================
CXXFLAGS := -std=c++20 -g -Wall -Wextra \
            `pkg-config --cflags glfw3` \
            -I$(VULKAN_INC) \
            -I$(GLM_INC) \
            -I$(STB_INC) \
            -I$(INCLUDE_DIR) \
            -I$(APP_DIR) \
            -DNDEBUG \
            $(PROFILING_FLAGS)

# ===============================
# Linker flags
# ===============================
LDFLAGS := `pkg-config --libs glfw3` -L$(VULKAN_LIB) -lvulkan

# ===============================
# Default target
# ===============================
all: $(TARGET)

# ===============================
# Link object files
# ===============================
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# ===============================
# Build rules
# ===============================
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/app_%.o: $(APP_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ===============================
# Clean build artifacts
# ===============================
clean:
	rm -rf $(BUILD_DIR) $(TARGET) profile_output.json

.PHONY: clean all

# ===============================
# Compile shaders
# ===============================
shaders:
ifeq ($(UNAME_S),Darwin)
	glslc -fshader-stage=vert shaders/vert.glsl -o shaders/vert.spv && \
	glslc -fshader-stage=frag shaders/frag.glsl -o shaders/frag.spv
else
	/usr/bin/glslc -fshader-stage=vert shaders/vert.glsl -o shaders/vert.spv && \
	/usr/bin/glslc -fshader-stage=frag shaders/frag.glsl -o shaders/frag.spv
endif

.PHONY: shaders

# ===============================
# Format with clang-format
# ===============================
FORMAT_DIR := src
CLANG_FORMAT := clang-format
FORMAT_STYLE := file

format:
	find $(FORMAT_DIR) -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -exec $(CLANG_FORMAT) -i -style=$(FORMAT_STYLE) {} +

.PHONY: format

# ===============================
# Generate docs
# ===============================
docs:
	doxygen Doxyfile

.PHONY: docs
