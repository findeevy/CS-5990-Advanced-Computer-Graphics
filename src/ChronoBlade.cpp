#include "ChronoBlade.hpp"
#include <iostream>
#include <iomanip>

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
 */
void ChronoBlade::end(const std::string& section) {
    auto endTime = std::chrono::high_resolution_clock::now();
    auto startTime = times[section];
    double ms = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    results[section].first += ms;
    results[section].second += 1;
}

/**
 * @brief Print a report of all recorded sections.
 * @details Loops over all results and prints each section's name, average time,
 *          and number of calls to std::cout.
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
                  << ": avg " << avg << " ms over " << data.second << " calls\n";
    }
}
