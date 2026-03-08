#ifndef EVENT_H
#define EVENT_H

#include <vector>
#include <string>
#include <cstdint>

// A single event from a Prophesee event camera.
//
// Each event says "pixel (x,y) changed brightness at time t with polarity p".
// ON events (p=1) mean the pixel got brighter, OFF events (p=0) mean darker.
// Timestamps are stored in seconds here (the raw H5 data is in microseconds).
struct Event {
    double   t;     // timestamp (seconds)
    uint16_t x;     // pixel column
    uint16_t y;     // pixel row
    int8_t   p;     // polarity: 1=ON (brighter), 0=OFF (darker)

    Event() : t(0), x(0), y(0), p(0) {}
    Event(double t, uint16_t x, uint16_t y, int8_t p)
        : t(t), x(x), y(y), p(p) {}
};

// A batch of events from a time window, plus the sensor dimensions.
struct EventBatch {
    std::vector<Event> events;
    uint16_t width  = 0;
    uint16_t height = 0;

    double startTime() const { return events.empty() ? 0.0 : events.front().t; }
    double endTime()   const { return events.empty() ? 0.0 : events.back().t;  }
    double duration()  const { return endTime() - startTime(); }
    size_t size()      const { return events.size(); }
    bool   empty()     const { return events.empty(); }
};

// Reads Prophesee event data from an M3ED-format HDF5 file.
//
// The H5 layout we expect:
//   prophesee/<side>/t            int64   timestamps in microseconds
//   prophesee/<side>/x            uint16  column
//   prophesee/<side>/y            uint16  row
//   prophesee/<side>/p            int8    polarity
//   prophesee/<side>/ms_map_idx   uint64  index of first event per millisecond
//   prophesee/<side>/calib/resolution  int64[2]  (width, height)
//
// The ms_map_idx table is the key trick: it tells us where each millisecond
// starts in the event array, so we can slice any time window without scanning
// through billions of timestamps.
class H5EventReader {
public:
    bool open(const std::string& path, const std::string& side = "left");

    uint16_t width()  const { return width_;  }
    uint16_t height() const { return height_; }
    size_t   totalEvents() const { return total_events_; }
    double   totalDurationSec() const;

    // Read all events in the time window [t0, t0+duration) seconds.
    bool readBatch(double t0_sec, double duration_sec, EventBatch& batch);

    // Read events by raw index range [begin, end).
    bool readRange(size_t idx_begin, size_t idx_end, EventBatch& batch);

private:
    std::string path_;
    std::string group_;                      // "prophesee/left" or "prophesee/right"
    uint16_t width_  = 0;
    uint16_t height_ = 0;
    size_t total_events_ = 0;
    int64_t t0_us_ = 0;                     // first timestamp in µs
    std::vector<uint64_t> ms_map_idx_;       // per-millisecond event index
};

#endif // EVENT_H
