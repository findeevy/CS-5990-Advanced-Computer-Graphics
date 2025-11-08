#include "ChronoProfiler.hpp"

/**
 * @file ChronoProfiler.cpp
 * @brief Implementation of lightweight CPU profiler declared in ChronoProfiler.hpp.
 *
 * @details
 * This file contains the logic for the ChronoProfiler:
 *
 *  - Records zone begin / end times using pushEventStart() and pushEventEnd()
 *  - Uses thread-local storage so events are collected independently per thread
 *  - During endFrame(), merges events into a shared frame event list for visualization
 *
 * The profiler is intentionally lightweight:
 *  - No dynamic allocation aside from vector growth during initial frames
 *  - No locks on the hot path (only at frame merge)
 *  - Suitable for real-time engines (graphics, simulation, game loops)
 */

// =============================================================================
//   STATIC STORAGE — ONE INSTANCE SHARED ACROSS PROGRAM
// =============================================================================
// These static members must be defined in exactly one .cpp file.

/// Thread-local event list — each thread pushes into its own isolated vector.
/// This avoids lock contention and keeps overhead extremely low.
thread_local std::vector<ChronoProfiler::Event> ChronoProfiler::threadEvents;

/// Final merged event list for the completed frame.
/// All thread-local event vectors are appended into this at endFrame().
std::vector<ChronoProfiler::Event> ChronoProfiler::frameEvents;

/// Mutex guarding merging of thread events. No locking occurs during pushEventStart()/pushEventEnd().
std::mutex ChronoProfiler::mergeMutex;

/// Time point indicating the moment beginFrame() was called.
/// Used to calculate relative timestamps (timeline visualization needs offsets).
std::chrono::high_resolution_clock::time_point ChronoProfiler::frameStart;

// =============================================================================
//   INTERNAL HELPER — TIME IN MILLISECONDS
// =============================================================================

/**
 * @brief Get current timestamp in milliseconds.
 *
 * @details
 * Converts the current time since epoch to milliseconds using high_resolution_clock.
 *
 * @return Current timestamp in milliseconds (double precision).
 */
double ChronoProfiler::nowMs() {
    using namespace std::chrono;

    // Convert time_since_epoch() -> milliseconds as floating point
    return duration<double, std::milli>(
            high_resolution_clock::now().time_since_epoch()
    ).count();
}

// =============================================================================
//   FRAME LIFECYCLE — beginFrame() / endFrame()
// =============================================================================

/**
 * @brief Start a new profiling frame.
 *
 * @details
 * This function must be called once per frame (usually at the start of the render loop).
 * It captures a reference time and clears the previous frame's merged event list.
 *
 * Example usage:
 * @code
 * ChronoProfiler::beginFrame();
 * PROFILE_SCOPE("Render");
 * drawFrame();
 * ChronoProfiler::endFrame();
 * @endcode
 */
void ChronoProfiler::beginFrame() {
    // Record when this frame began.
    frameStart = std::chrono::high_resolution_clock::now();

    // Ensure no stale data leaks across frames.
    frameEvents.clear();
}

/**
 * @brief Finalize profiling for the frame.
 *
 * @details
 * Thread-local event buffers are merged into a single shared event list.
 * This is the ONLY place we perform locking (mergeMutex).
 *
 * Visualization (e.g., ImGui timeline viewer) consumes the data returned by getEvents().
 */
void ChronoProfiler::endFrame() {
    std::lock_guard<std::mutex> lock(mergeMutex);

    // Append all thread-local events into the global frame buffer.
    frameEvents.insert(frameEvents.end(), threadEvents.begin(), threadEvents.end());

    // Reset per-thread storage for next frame.
    threadEvents.clear();
}

// =============================================================================
//   PROFILING ZONE START / END — pushEventStart() / pushEventEnd()
// =============================================================================

/**
 * @brief Begin a profiling zone.
 *
 * @details
 * Called internally by `PROFILE_SCOPE(name)` when entering a scope.
 *
 * This function:
 *  - Stores timestamp (absolute time)
 *  - Saves thread ID (converted to uint32_t)
 *  - Defers duration calculation until pushEventEnd()
 *
 * @param name Name of the profiling zone (caller-owned string literal recommended).
 */
void ChronoProfiler::pushEventStart(const char* name) {
    Event evt = {};

    evt.name = name;

    // Store timestamp in absolute time (converted to relative later).
    evt.startMs = nowMs();

    // Duration unknown until pushEventEnd().
    evt.durationMs = -1.0;

    // Convert opaque std::thread::id into numeric hash (consistent per thread lifetime).
    evt.threadId = static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));

    // Put this event into THIS thread's TLS event buffer.
    threadEvents.push_back(evt);
}

/**
 * @brief End a profiling zone.
 *
 * @details
 * Matches with a previous pushEventStart().
 * Duration is computed, and startMs is offset relative to frame start.
 */
void ChronoProfiler::pushEventEnd() {
    // Safety: end call without a matching start (should not happen).
    if (threadEvents.empty())
        return;

    // Retrieve the event currently being finalized.
    Event& evt = threadEvents.back();

    // Capture end time in ms.
    double endTimeMs = nowMs();

    // Duration = end - absolute start
    evt.durationMs = endTimeMs - evt.startMs;

    // Convert start timestamp into "ms since beginFrame()"
    evt.startMs -= std::chrono::duration<double, std::milli>(frameStart.time_since_epoch()).count();
}

// =============================================================================
//   EVENT ACCESSOR — getEvents()
// =============================================================================

/**
 * @brief Retrieve finalized events recorded during the previous frame.
 *
 * @return Reference to the vector containing all finalized events.
 *
 * @warning Do not store the returned reference beyond the current frame.
 */
const std::vector<ChronoProfiler::Event>& ChronoProfiler::getEvents() {
    return frameEvents;
}
