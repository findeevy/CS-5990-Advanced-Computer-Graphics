#include "ChronoBlade.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>

/**
 * @brief Start timing a section.
 * @param section Name of the section.
 * @details Records the current timestamp in `times[section]`.
 */
void ChronoBlade::start(const std::string& section) {
    times[section] = std::chrono::high_resolution_clock::now();
}

/**
 * @brief End timing a section.
 * @param section Name of the section previously started.
 * @details Computes elapsed time in milliseconds since `start()` and
 *          accumulates it in `results[section].first`, while incrementing
 *          the call count in `results[section].second`.
 *
 * @warning If `start()` was never called for this section, this function
 *          prints a warning and ignores the call.
 */
void ChronoBlade::end(const std::string& section) {
    auto it = times.find(section);
    if (it == times.end()) {
        std::cerr << "[ChronoBlade WARNING] end() called before start(): "
                  << section << "\n";
        return;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(endTime - it->second).count();

    results[section].first  += ms;
    results[section].second += 1;
}

/**
 * @brief Print a report of all recorded sections to stdout.
 *
 * @details Loops over all results and prints each section's:
 *   - name
 *   - average time (ms)
 *   - number of calls
 *
 * Example output:
 * @code
 * --- ChronoBlade Report ---
 * drawFrame                     : avg 16.7 ms over 300 calls
 * recordCommandBuffer           : avg 2.3 ms over 300 calls
 * @endcode
 */
void ChronoBlade::report() const {
    std::cout << "\n--- ChronoBlade Report ---\n";

    for (const auto& [name, data] : results) {
        double avg = data.first / data.second;

        std::cout << std::setw(30) << std::left << name
                  << ": avg " << avg << " ms over "
                  << data.second << " calls\n";
    }
}

/**
 * @brief Export profiling results to a CSV file.
 *
 * @param filename Name of the output CSV file (ex: "profile.csv")
 *
 * @details Output format:
 * @code
 * section,avg_ms,calls
 * drawFrame,16.7,300
 * recordCommandBuffer,2.3,300
 * @endcode
 *
 * @throws std::runtime_error if file cannot be opened.
 */
void ChronoBlade::exportCSV(const std::string& filename) const {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("ChronoBlade: Failed to write CSV to " + filename);
    }

    file << "section,avg_ms,calls\n";

    for (const auto& [name, data] : results) {
        double avg = data.first / data.second;
        file << name << "," << avg << "," << data.second << "\n";
    }

    file.close();
    std::cout << "[ChronoBlade] CSV export complete → " << filename << "\n";
}
