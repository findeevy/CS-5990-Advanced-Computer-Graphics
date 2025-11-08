#pragma once

/**
 * @file ChronoProfiler.hpp
 * @brief Header declaration for a lightweight CPU profiler with zone/timeline capture.
 *
 * This file only contains declarations. Implementation lives in ChronoProfiler.cpp
 */

//
// ──────────────────────────────────────────────────────────────────────────────
//   Includes — Each one explained
// ──────────────────────────────────────────────────────────────────────────────
//
#include <chrono>        ///< Provides high-resolution timing (std::chrono::high_resolution_clock)
#include <string>        ///< Allows passing zone names as const char*/string
#include <vector>        ///< Stores profiling events in per-thread and per-frame buffers
#include <thread>        ///< Used to identify the current thread (std::this_thread::get_id)
#include <mutex>         ///< Protects merging events from TLS into a frame buffer
#include <unordered_map> ///< (optional in future) could map thread names to IDs

/**
 * @class ChronoProfiler
 * @brief Real-time, zone-based CPU profiler with per-thread event timelines.
 *
 * The profiler allows you to mark regions of code using the PROFILE_SCOPE(name) macro:
 *
 * @code
 * void update() {
 *     PROFILE_SCOPE("Update Logic");
 *     expensiveCall();
 * }
 * @endcode
 *
 * Events are stored per-thread using thread-local buffers and merged at endFrame().
 * Visualization (imgui, json exporter, etc.) is handled outside of this class.
 */
class ChronoProfiler {
public:
    /**
     * @struct Event
     * @brief Represents a profiled execution zone within a single frame.
     *
     * Each `Event` is a single timing measurement for a named zone.
     */
    struct Event {
        const char* name;   ///< Human-readable label for the zone (caller-owned string)
        double startMs;     ///< Timestamp in milliseconds relative to frame start
        double durationMs;  ///< Duration of the zone in milliseconds
        uint32_t threadId;  ///< Numeric ID of the thread that captured this event
    };

    /**
     * @brief Marks the beginning of a new frame of profiling.
     *
     * This resets per-frame buffers and records a new reference start time so
     * all subsequent events are relative to this moment.
     */
    static void beginFrame();

    /**
     * @brief Finalizes the frame.
     *
     * All thread-local events recorded by PROFILE_SCOPE during the frame
     * are merged into a single `frameEvents` list.
     */
    static void endFrame();

    /**
     * @brief Capture the start of a profiling zone.
     *
     * Intended for internal use — use PROFILE_SCOPE instead for RAII behavior.
     *
     * @param name The name of the profiled scope.
     */
    static void pushEventStart(const char* name);

    /**
     * @brief Capture the completion of a profiling zone.
     *
     * Calculates the duration and stores a finalized Event entry.
     */
    static void pushEventEnd();

    /**
     * @brief Returns the list of finalized events for the previous frame.
     *
     * @return Reference to the vector containing completed Event entries.
     */
    static const std::vector<Event>& getEvents();

    /**
     * @class ScopedZone
     * @brief RAII helper for PROFILE_SCOPE(name)
     *
     * Constructs a scope profiler that automatically pushes a start event,
     * and pushes an end event when it destructs.
     */
    class ScopedZone {
    public:
        /**
         * @brief Start a new profiling zone.
         * @param name Label for the zone.
         */
        explicit ScopedZone(const char* name) { ChronoProfiler::pushEventStart(name); }

        /**
         * @brief Finalize profiling zone on scope exit.
         */
        ~ScopedZone() { ChronoProfiler::pushEventEnd(); }
    };

    /**
     * @class ScopedFrame
     * @brief RAII helper for profiling an entire frame.
     *
     * Constructs a frame profiler that automatically calls ChronoProfiler::beginFrame()
     * on construction, and ChronoProfiler::endFrame() when it destructs.
     *
     * Usage:
     * @code
     * void mainLoop() {
     *     while (!glfwWindowShouldClose(window)) {
     *         glfwPollEvents();
     *
     *         ScopedFrame frame;          // automatically begins frame
     *         PROFILE_SCOPE("drawFrame"); // zone profiling inside frame
     *         drawFrame();
     *     } // frame destructor automatically ends the frame
     * }
     * @endcode
     */
    class ScopedFrame {
    public:
        /**
         * @brief Start a new profiling frame.
         */
        ScopedFrame() { ChronoProfiler::beginFrame(); }

        /**
         * @brief Finalize profiling frame on scope exit.
         */
        ~ScopedFrame() { ChronoProfiler::endFrame(); }
    };

private:
    //
    // ────────────────────────────────────────────────────────────
    //   Internal State (owned by ChronoProfiler)
    // ────────────────────────────────────────────────────────────
    //

    /// Thread-local list of events **recorded from this thread only**
    static thread_local std::vector<Event> threadEvents;

    /// Final merged events for a frame (gathered from all threads)
    static std::vector<Event> frameEvents;

    /// Prevents race conditions during event merge
    static std::mutex mergeMutex;

    /// Time point representing when the current frame began
    static std::chrono::high_resolution_clock::time_point frameStart;

    /**
     * @brief Utility function to get current time in milliseconds.
     *
     * @return Timestamp relative to epoch, as ms.
     */
    static double nowMs();
};

/**
 * @def PROFILE_SCOPE(name)
 * @brief Automatically profiles a scope using RAII.
 *
 * Usage:
 * @code
 * PROFILE_SCOPE("Physics Update");
 * @endcode
 */
#define PROFILE_SCOPE(name) ChronoProfiler::ScopedZone _scope_##__LINE__(name)
