#include "opt-sched/Scheduler/cuda_lnkd_lst.cuh"
#include "opt-sched/Scheduler/sched_basic_data.h"

namespace llvm {
namespace opt_sched {

template <class T> inline
__host__ __device__
T *LinkedList<T>::GetFrstElmnt(){
  printf("Inside GetFrstElmnt\n");
  wasTopRmvd_ = false;
  wasBottomRmvd_ = false;
  rtrvEntry_ = topEntry_;
  return rtrvEntry_ == NULL ? NULL : rtrvEntry_->element;
}

template <class T> inline 
__host__ __device__
T *LinkedList<T>::GetNxtElmnt() {
  printf("Inside GetNxtElmnt\n");
  if (wasTopRmvd_) {
    rtrvEntry_ = topEntry_;
  } else {
    rtrvEntry_ = rtrvEntry_->GetNext();
  }

  if (wasBottomRmvd_) {
    rtrvEntry_ = NULL;
  }

  wasTopRmvd_ = false;
  wasBottomRmvd_ = false;
  T *elmnt = rtrvEntry_ == NULL ? NULL : rtrvEntry_->element;
  return elmnt;
}

template SchedInstruction* LinkedList<SchedInstruction>::GetFrstElmnt<SchedInstruction>();
template SchedInstruction* LinkedList<SchedInstruction>::GetNxtElmnt<SchedInstruction>();
}
}
