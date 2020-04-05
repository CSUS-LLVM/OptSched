/*******************************************************************************
Description:  Implements a simple logger that writes out messages to a file or
              to a standard stream.
Author:       Max Shawabkeh
Created:      Mar. 2011
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_GENERIC_LOGGER_H
#define OPTSCHED_GENERIC_LOGGER_H

#include "opt-sched/Scheduler/defines.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <type_traits>
#include <utility>

namespace llvm {
namespace opt_sched {

namespace Logger {
// Error severity levels.
enum LOG_LEVEL {
  // Fatal error. Exit program.
  FATAL = 1,
  // Non-fatal error. Program should continue.
  ERROR = 2,
  // Generic non-error logging message.
  INFO = 4,
  // A summary message that should be shown only in the summary log.
  SUMMARY = 8
};

// Directs all subsequent log output to the specified output stream. Defaults
// to the standard error stream if not set.
void SetLogStream(std::ostream &out);
std::ostream &GetLogStream();

// Output a log message of a given level, either with a timestamp or without.
// Expects a printf-style format string and a variable number of arguments to
// place into the string.
void Log(LOG_LEVEL level, bool timed, const char *format_string, ...);

// Registers a periodic logging function that will respond to being called at
// most every period milliseconds and act as a no-op until the period has
// passed. Note that time measuring is in process CPU time.
void RegisterPeriodicLogger(Milliseconds period, void (*callback)());
// Runs the previously registered logging function. If the period has not
// passed since the last call to PeriodicLog() or RegisterPeriodicLogger(),
// this acts as a no-op.
void PeriodicLog();

// Shortcuts for each logging level.
[[noreturn]] void Fatal(const char *format_string, ...);
void Error(const char *format_string, ...);
void Info(const char *format_string, ...);
void Summary(const char *format_string, ...);

namespace detail {
// TODO: When we get C++17, get rid of EventAttrType and EventAttrValue in favor
// of a std::variant.

/** Encodes the type of an Event attribute */
enum class EventAttrType {
  Int64,
  UInt64,
  CStr,
  Bool,
};

/* Gets the type of the argument */
inline EventAttrType GetEventAttrType(const char *) {
  return EventAttrType::CStr;
}

inline EventAttrType GetEventAttrType(bool) { return EventAttrType::Bool; }

template <typename Int,
          typename std::enable_if<std::is_integral<Int>::value, int>::type = 0>
inline EventAttrType GetEventAttrType(Int) {
  // Treat anything which is not a uint64_t as an int64_t.
  // This may aid branch prediction in the implementation.
  return (std::is_signed<Int>::value || sizeof(Int) < sizeof(int64_t))
             ? EventAttrType::Int64
             : EventAttrType::UInt64;
}

/** Encodes the value of an Event attribute. */
union EventAttrValue {
  int64_t i64;
  uint64_t u64;
  const char *cstr;
  bool b;

  EventAttrValue(const char *val) : cstr{val} {}
  EventAttrValue(bool val) : b{val} {}

  template <typename Int, typename std::enable_if<std::is_integral<Int>::value,
                                                  int>::type = 0>
  EventAttrValue(Int val) {
    if (std::is_signed<Int>::value || sizeof(Int) < sizeof(int64_t)) {
      i64 = val;
    } else {
      u64 = val;
    }
  }
};

/** The implementation of Logger::Event(...) */
void Event(const std::pair<EventAttrType, EventAttrValue> *attrs,
           size_t numAttrs);
} // namespace detail

/**
 * \brief Logs an event in a json format.
 * \detail
 *
 * ``Logger::Event(eventID, [key, value]...)``
 *
 * Logs messages of the format `EVENT: {"event_id": eventID, "key": value...}`,
 * allowing for easier parsing by tools later down the line. The current time is
 * always included.
 *
 * \param eventID a unique ID identifying this event. This should match the
 * regular expression `[A-Z0-9_]+`. That is, this should contain no spaces.
 *
 * \param args An alternating list of keys and values.
 *
 * \warning Any change to a log statement of this format requires a change in
 * our log-parsing scripts.
 */
template <typename... Args>
void Event(const char *eventID, const Args &... args) {
  static_assert(sizeof...(args) % 2 == 0,
                "Every key must have a corresponding value.");

  using EventItem = std::pair<detail::EventAttrType, detail::EventAttrValue>;

  std::array<EventItem, sizeof...(args) + 2> arr{
      EventItem(detail::EventAttrType::CStr,
                detail::EventAttrValue("event_id")),
      EventItem(detail::EventAttrType::CStr, detail::EventAttrValue(eventID)),
      EventItem(detail::GetEventAttrType(args),
                detail::EventAttrValue(args))...,
  };

  detail::Event(arr.data(), arr.size());
}

} // namespace Logger

} // namespace opt_sched
} // namespace llvm

#endif
