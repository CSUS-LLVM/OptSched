#include "opt-sched/Scheduler/utilities.h"
#include <chrono>

using namespace llvm::opt_sched;

std::chrono::steady_clock::time_point Utilities::startTime =
    std::chrono::steady_clock::now();
