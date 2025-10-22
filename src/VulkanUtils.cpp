#include "VulkanUtils.hpp"

std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    std::cout << filename << std::endl;

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file!");
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}
