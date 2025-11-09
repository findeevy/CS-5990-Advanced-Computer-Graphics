#include "ProfilerUI.hpp"

/**
 * @file ProfilerUI.cpp
 * @brief Implementation of the ProfilerUI class for console-based visualization
 * of ChronoProfiler events.
 */

/**
 * @brief Construct a new ProfilerUI object
 * @param historySize Maximum number of frames to store in rolling history
 */
ProfilerUI::ProfilerUI(size_t historySize)
        : maxHistory(historySize) { }

/**
 * @brief Update the UI with the latest frame events
 *
 * This function should be called once per frame after ChronoProfiler::endFrame().
 * It copies the merged frame events into the rolling history, maintains history size,
 * and updates aggregated statistics per zone.
 */
void ProfilerUI::update() {
    std::lock_guard<std::mutex> lock(uiMutex);  ///< Ensure thread-safety

    // Get merged events from the last frame
    const auto& events = ChronoProfiler::getEvents();

    // Add the new frame to history
    frameHistory.push_back(events);

    // Maintain rolling history by removing oldest frame if exceeded max
    if (frameHistory.size() > maxHistory) {
        frameHistory.erase(frameHistory.begin());
    }

    // Update aggregated statistics per zone
    for (const auto& e : events) {
        aggregatedStats[e.name].add(e.durationMs);
    }
}

/**
 * @brief Render the profiler UI to the console
 *
 * Prints:
 * - Frame-specific zone breakdown with ASCII bars
 * - Aggregated statistics across all frames
 */
void ProfilerUI::render() {
    std::lock_guard<std::mutex> lock(uiMutex);  ///< Ensure thread-safety

    size_t frameIndex = frameHistory.size();
    std::cout << "\n=== Frame " << frameIndex << " ===\n";

    // Render last frame events as ASCII bars
    if (!frameHistory.empty()) {
        renderFrame(frameHistory.back(), frameIndex);
    }

    // Render aggregated statistics table
    renderAggregatedStats();
}

/**
 * @brief Render a single frame's events as ASCII bars
 *
 * @param events Vector of ChronoProfiler::Event for the frame
 * @param frameIndex Index of the frame (for labeling)
 *
 * Each zone prints:
 * - Name left-aligned
 * - Duration represented as "█" characters scaled by 10x
 * - Numeric duration in milliseconds
 * - Thread name
 */
void ProfilerUI::renderFrame(const std::vector<ChronoProfiler::Event>& events, size_t frameIndex) {
    for (const auto& e : events) {
        // Scale duration for ASCII bar (adjust 10x for better visibility)
        int barLength = static_cast<int>(e.durationMs * 10);

        // Print zone name
        std::cout << std::setw(20) << std::left << e.name << " ";

        // Print ASCII bar
        for (int i = 0; i < barLength; ++i) std::cout << "█";

        // Print numeric duration and thread name
        std::cout << " " << std::fixed << std::setprecision(2) << e.durationMs << " ms";
        std::cout << " [" << e.threadName << "]\n";
    }
}

/**
 * @brief Render aggregated statistics for all zones
 *
 * Displays a table with:
 * - Zone name
 * - Average duration across frames
 * - Maximum duration observed
 * - Number of occurrences
 */
void ProfilerUI::renderAggregatedStats() {
    std::cout << "\n-- Aggregated Stats --\n";
    std::cout << std::setw(20) << "Zone"
              << std::setw(10) << "Avg(ms)"
              << std::setw(10) << "Max(ms)"
              << std::setw(10) << "Count\n";

    for (const auto& [name, stats] : aggregatedStats) {
        std::cout << std::setw(20) << name
                  << std::setw(10) << std::fixed << std::setprecision(2) << stats.avg()
                  << std::setw(10) << stats.maxMs
                  << std::setw(10) << stats.count << "\n";
    }
}
