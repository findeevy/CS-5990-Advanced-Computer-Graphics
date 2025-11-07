#pragma once
#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>
#include <iomanip>

/**
 * @class ChronoBlade
 * @brief High-resolution profiler for named code sections with CSV export support.
 *
 * @details
 * ChronoBlade allows developers to profile execution time of multiple parts of
 * their program by assigning a string name to each instrumented region.
 *
 * Usage pattern:
 *   - Call `start("sectionName")` at the beginning of a timed section.
 *   - Call `end("sectionName")` at the end of the same section.
 *   - Call `report()` to show results in console.
 *   - Call `exportCSV("file.csv")` to save profiling results to a CSV file.
 *
 * @note This profiler is intended for real-time applications (e.g., Vulkan render loops).
 *       It has minimal overhead, but avoid using it inside inner loops that execute millions
 *       of times per frame.
 *
 * Example usage:
 * @code
 * ChronoBlade profiler;
 * profiler.start("loadResources");
 * // load textures, models, etc.
 * profiler.end("loadResources");
 *
 * profiler.report();              // prints formatted terminal report
 * profiler.exportCSV("profile.csv"); // enables graphing / visualization
 * @endcode
 *
 * @author Eric Newton
 * @version 1.1
 * @date 2025-11-07
 */
class ChronoBlade {
public:
    /**
     * @brief Start timing a section.
     *
     * @param section Name of the section to start timing.
     *
     * @details Stores the current high-resolution timestamp. If a section is started twice
     *          without calling `end()`, the previous start time is overwritten.
     */
    void start(const std::string& section);

    /**
     * @brief End timing a section and accumulate results.
     *
     * @param section Name of the section previously passed to `start()`.
     *
     * @details Computes elapsed time since the last call to `start(section)` and adds
     *          it to the cumulative timing record. Also increments the call count.
     *
     * @warning Calling `end()` without a prior `start()` for the same section results in
     *          a warning and the call will be ignored.
     */
    void end(const std::string& section);

    /**
     * @brief Print formatted timing statistics to stdout.
     *
     * @details Generates a formatted list showing:
     *   - Section name
     *   - Average time (ms)
     *   - Call count
     *
     * Useful for debugging performance inside terminal-based workflows.
     */
    void report() const;

    /**
     * @brief Export profiling results to a CSV file.
     *
     * @param filename Output CSV filename (e.g., `"metrics.csv"`).
     *
     * @details Creates a CSV file with columns:
     *   section,avg_ms,calls
     *
     * This makes it extremely easy to visualize results in:
     *   - Excel / Google Sheets
     *   - Python (matplotlib, seaborn)
     *   - Grafana / web dashboards
     *
     * @throws std::runtime_error If file cannot be opened.
     */
    void exportCSV(const std::string& filename) const;

private:
    /// Start timestamps for active sections
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> times;

    /// Accumulated time (ms) and call count per section
    std::unordered_map<std::string, std::pair<double, int>> results;
};
