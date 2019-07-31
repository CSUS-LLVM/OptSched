#include "opt-sched/Scheduler/logger.h"
// For va_list, va_start(), va_end().
#include <cstdarg>
// For sprintf(), vsprintf().
#include <cstdio>
// For exit().
#include <cstdlib>
// For GetProcessorTime().
#include "opt-sched/Scheduler/utilities.h"

using namespace llvm::opt_sched;

// An ugly macro to simplify repeated vararg-insertion.
#define VPRINT(buf, frmt)                                                      \
  va_list args;                                                                \
  va_start(args, frmt);                                                        \
  vsprintf(buf, frmt, args);                                                   \
  va_end(args);

// The maximum buffer size for error messages.
static const int MAX_MSGSIZE = 8000;

// The current output stream.
static std::ostream *logStream = &std::cerr;

// The periodic logging callback.
static void (*periodLogCallback)() = NULL;
// The minimum length of (CPU) time between two calls to the periodic logging
// callback.
static Milliseconds periodLogPeriod = 0;
// The CPU time when the period log was last called.
static Milliseconds periodLogLastTime = 0;

// The main output function. Calculates the time since process start and formats
// the specified message with a title and timestamp. Exits the program with exit
// code = 1 on fatal errors.
static void Output(Logger::LOG_LEVEL level, bool timed, const char *message) {
  const char *title = 0;

  switch (level) {
  case Logger::FATAL:
    title = "FATAL";
    break;
  case Logger::ERROR:
    title = "ERROR";
    break;
  case Logger::INFO:
    title = "INFO";
    break;
  case Logger::SUMMARY:
    title = "SUMMARY";
    break;
  }

  (*logStream) << title << ": " << message;
  if (timed) {
    (*logStream) << " (Time = " << Utilities::GetProcessorTime() << " ms)";
  }
  (*logStream) << std::endl;

  if (level == Logger::FATAL)
    exit(1);
}

void Logger::SetLogStream(std::ostream &out) { logStream = &out; }

std::ostream &Logger::GetLogStream() { return *logStream; }

void Logger::RegisterPeriodicLogger(Milliseconds period, void (*callback)()) {
  periodLogLastTime = Utilities::GetProcessorTime();
  periodLogCallback = callback;
  periodLogPeriod = period;
}

void Logger::PeriodicLog() {
  if (!periodLogCallback) {
    Error("Periodic log called while no callback was registered.");
    return;
  }

  Milliseconds now = Utilities::GetProcessorTime();
  ;
  if (now - periodLogLastTime >= periodLogPeriod) {
    periodLogCallback();
    periodLogLastTime = now;
  }
}

void Logger::Log(Logger::LOG_LEVEL level, bool timed, const char *format_string,
                 ...) {
  char message_buffer[MAX_MSGSIZE];
  VPRINT(message_buffer, format_string);
  Output(level, timed, message_buffer);
}

void Logger::Fatal(const char *format_string, ...) {
  char message_buffer[MAX_MSGSIZE];
  VPRINT(message_buffer, format_string);
  Output(Logger::FATAL, true, message_buffer);
}

void Logger::Error(const char *format_string, ...) {
  char message_buffer[MAX_MSGSIZE];
  VPRINT(message_buffer, format_string);
  Output(Logger::ERROR, true, message_buffer);
}

void Logger::Info(const char *format_string, ...) {
  char message_buffer[MAX_MSGSIZE];
  VPRINT(message_buffer, format_string);
  Output(Logger::INFO, true, message_buffer);
}

void Logger::Summary(const char *format_string, ...) {
  char message_buffer[MAX_MSGSIZE];
  VPRINT(message_buffer, format_string);
  Output(Logger::SUMMARY, false, message_buffer);
}
