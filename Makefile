CXX = clang++
# Remove -g to run in a non-debug mode.
CXXFLAGS = -std=c++20 -g -Wall -Wextra `pkg-config --cflags glfw3` -I$(VULKAN_SDK)/include
LDFLAGS = `pkg-config --libs glfw3` -L$(VULKAN_SDK)/lib -lvulkan -rpath $(VULKAN_SDK)/lib

# macOS-specific Vulkan/MoltenVK settings
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # Use MoltenVK framework on macOS
    LDFLAGS = `pkg-config --libs glfw3` -framework Metal -framework Foundation -framework IOKit -framework QuartzCore -L$(VULKAN_SDK)/lib -lMoltenVK -lvulkan -rpath $(VULKAN_SDK)/lib
    # Alternative if using MoltenVK from Homebrew
    # LDFLAGS = `pkg-config --libs glfw3` -framework Metal -framework Foundation -framework IOKit -framework QuartzCore -L/usr/local/lib -lMoltenVK
endif

TARGET = CS5990 
SRCS = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

shaders:
	/usr/bin/glslc -fshader-stage=vert shaders/vert.glsl -o shaders/vert.spv && /usr/bin/glslc -fshader-stage=frag shaders/frag.glsl -o shaders/frag.spv

.PHONY: shaders

FORMAT_EXTENSIONS := *.cpp *.h
FORMAT_DIR := src
CLANG_FORMAT := clang-format
FORMAT_STYLE := file

format:
	find $(FORMAT_DIR) -type f \( -name "*.cpp" -o -name "*.h" \) -exec $(CLANG_FORMAT) -i -style=$(FORMAT_STYLE) {} +

.PHONY: format
