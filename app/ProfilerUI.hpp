#pragma once

/**
 * @file ProfilerUI.hpp
 * @brief Console-based UI for visualizing ChronoProfiler results.
 *
 * This file provides a simple, homemade profiler visualization layer that does not depend
 * on external GUI libraries. It prints ASCII bars for zone durations and aggregates
 * statistics over multiple frames.
 */

#include <ChronoProfiler.hpp>      // For access to ChronoProfiler::Event and frame events
#include <vector>                  // Used for storing events per frame
#include <string>                  // Used for zone names and thread names
#include <unordered_map>           // Used to aggregate statistics per zone
#include <mutex>                   // Protects UI updates in multithreaded context
#include <iostream>                // Used for printing to console
#include <iomanip>                 // Used for formatting console output (width, precision)

/**
 * @struct ZoneStats
 * @brief Stores aggregated statistics for a profiling zone across frames.
 *
 * This includes total time, maximum time, and number of occurrences for computing averages.
 */
struct ZoneStats {
    double totalMs = 0.0; ///< Total accumulated duration in milliseconds
    double maxMs = 0.0;   ///< Maximum duration encountered for this zone
    size_t count = 0;     ///< Number of events recorded for this zone

    /**
     * @brief Add a duration measurement to the statistics
     * @param duration The duration of a profiling zone in milliseconds
     */
    void add(double duration) {
        totalMs += duration;
        if (duration > maxMs) maxMs = duration;
        count++;
    }

    /**
     * @brief Compute the average duration for this zone
     * @return Average duration in milliseconds
     */
    double avg() const { return count ? totalMs / count : 0.0; }
};

/**
 * @class ProfilerUI
 * @brief Simple console-based UI to visualize ChronoProfiler events.
 *
 * This class maintains a rolling history of frames, displays ASCII bars for
 * zone durations, and computes aggregated statistics for each zone.
 */
class ProfilerUI {
public:

    /**
     * @brief Construct a new ProfilerUI object
     * @param historySize Maximum number of frames to keep in rolling history (default 60)
     */
    ProfilerUI(size_t historySize = 60);

    /**
     * @brief Update the UI with the latest frame events
     *
     * Should be called every frame, after ChronoProfiler::endFrame().
     * Merges thread-local events into the rolling history and updates aggregate stats.
     */
    void update();

    /**
     * @brief Render the current frame's profile breakdown and aggregated statistics
     *
     * Prints to console:
     * - Per-zone ASCII bar graph for last frame
     * - Aggregated average/max/count for all zones
     */
    void render();

private:
    size_t maxHistory; ///< Maximum number of frames in rolling history

    /**
     * @brief Rolling history of frames
     * Each element is a vector of ChronoProfiler::Event representing a single frame
     */
    std::vector<std::vector<ChronoProfiler::Event>> frameHistory;

    /**
     * @brief Aggregated statistics for all zones
     * Key: zone name
     * Value: ZoneStats containing total/max/count
     */
    std::unordered_map<std::string, ZoneStats> aggregatedStats;

    std::mutex uiMutex; ///< Mutex to protect updates and rendering in multithreaded context

    /**
     * @brief Render a single frame's events as ASCII bars
     * @param events Vector of ChronoProfiler::Event for the frame
     * @param frameIndex Index of the frame (for labeling)
     *
     * Each event is printed with:
     * - Name of the zone
     * - Duration in ms
     * - Thread name
     * - ASCII bar proportional to duration
     */
    void renderFrame(const std::vector<ChronoProfiler::Event>& events, size_t frameIndex);

    /**
     * @brief Render aggregated statistics for all zones
     *
     * Displays a table with:
     * - Zone name
     * - Average duration in ms
     * - Maximum duration in ms
     * - Number of occurrences
     */
    void renderAggregatedStats();
};
