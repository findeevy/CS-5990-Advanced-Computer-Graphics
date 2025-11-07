#pragma once
#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>
#include <iomanip>

/**
 * @class ChronoBlade
 * @brief High-resolution profiler for named code sections.
 *
 * @details
 * ChronoBlade allows developers to measure the execution time of different
 * parts of their code by giving each section a unique name.
 * Each section is timed with `start()` and `end()` calls.
 * ChronoBlade accumulates total time and counts the number of calls for each section.
 * At the end of execution or periodically, `report()` prints a formatted summary.
 *
 * @note Designed for **debugging and performance profiling** in real-time applications.
 *       Minimal overhead, but avoid overusing in extremely tight loops.
 *
 * Example usage:
 * @code
 * ChronoBlade profiler;
 * profiler.start("loadResources");
 * // load textures, models, etc.
 * profiler.end("loadResources");
 * profiler.report();
 * @endcode
 *
 * @author Eric Newton
 * @version 1.0
 * @date 2025-11-07
 */
class ChronoBlade {
public:
    /**
     * @brief Start timing a section.
     * @param section Name of the section to start timing.
     * @details Records the current high-resolution timestamp associated with the section name.
     *          This timestamp will be used by `end()` to compute elapsed time.
     */
    void start(const std::string& section);

    /**
     * @brief End timing a section.
     * @param section Name of the section previously started.
     * @details Computes elapsed time since `start()` was called for this section.
     *          Accumulates total time and increments call count.
     * @warning Calling `end()` without a prior `start()` for the same section is undefined.
     */
    void end(const std::string& section);

    /**
     * @brief Print a report of all timed sections.
     * @details Outputs each section's name, average duration in milliseconds,
     *          and number of calls to std::cout in a neatly formatted table.
     */
    void report() const;

private:
    /// Start timestamps for active sections
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> times;

    /// Accumulated time (ms) and call counts for each section
    std::unordered_map<std::string, std::pair<double, int>> results;
};
