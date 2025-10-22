#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

/**
 * @brief Reads a binary file into a std::vector<char>.
 * @param filename Path to the file.
 * @return std::vector<char> containing the file contents.
 * @throws std::runtime_error if the file cannot be opened.
 */
std::vector<char> readFile(const std::string &filename);
