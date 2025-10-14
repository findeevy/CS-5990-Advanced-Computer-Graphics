CXX = clang++
TARGET = CS5990
SRCS = src/main.cpp

# Detect OS
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
  # macOS paths
  VULKAN_INC = $(VULKAN_SDK)/include
  GLM_INC = /opt/homebrew/include
else ifeq ($(UNAME_S),Linux)
  # Linux paths
  VULKAN_INC = /usr/include
  GLM_INC = /usr/include
endif

# Local include for stb
STB_INC = $(HOME)/include

CXXFLAGS = -std=c++20 -g -Wall -Wextra \
           `pkg-config --cflags glfw3` \
           -I$(VULKAN_INC) \
           -I$(GLM_INC) \
           -I$(STB_INC)

LDFLAGS = `pkg-config --libs glfw3` -lvulkan

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

shaders:
	/usr/bin/glslc -fshader-stage=vert shaders/vert.glsl -o shaders/vert.spv && \
	/usr/bin/glslc -fshader-stage=frag shaders/frag.glsl -o shaders/frag.spv

.PHONY: shaders clean

# Formatting
FORMAT_EXTENSIONS := *.cpp *.h
FORMAT_DIR := src
CLANG_FORMAT := clang-format
FORMAT_STYLE := file

format:
	find $(FORMAT_DIR) -type f \( -name "*.cpp" -o -name "*.h" \) -exec $(CLANG_FORMAT) -i -style=$(FORMAT_STYLE) {} +

.PHONY: format
