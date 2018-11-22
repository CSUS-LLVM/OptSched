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
#include <iostream>

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
void Fatal(const char *format_string, ...);
void Error(const char *format_string, ...);
void Info(const char *format_string, ...);
void Summary(const char *format_string, ...);
}

} // namespace opt_sched
} // namespace llvm

#endif
