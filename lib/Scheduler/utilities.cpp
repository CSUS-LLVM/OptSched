#include "opt-sched/Scheduler/utilities.h"
#include <chrono>

using namespace llvm::opt_sched;

std::chrono::high_resolution_clock::time_point Utilities::startTime =
    std::chrono::high_resolution_clock::now();
