#include "opt-sched/Scheduler/aco.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "llvm/ADT/STLExtras.h"
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace llvm::opt_sched;

#ifndef NDEBUG
static void PrintInstruction(SchedInstruction *inst);
#endif
void PrintSchedule(InstSchedule *schedule);

double RandDouble(double min, double max) {
  double rand = (double)RandomGen::GetRand32() / INT32_MAX;
  return (rand * (max - min)) + min;
}

#define USE_ACS 1
//#define BIASED_CHOICES 10000000
//#define LOCAL_DECAY 0.1

//#if USE_ACS
//#define ANTS_PER_ITERATION 10
//#define DECAY_FACTOR 0.1
//#else
//#define ANTS_PER_ITERATION count_
//#define DECAY_FACTOR 0.5
//#endif

ACOScheduler::ACOScheduler(DataDepGraph *dataDepGraph,
                           MachineModel *machineModel, InstCount upperBound,
                           SchedPriorities priorities, bool vrfySched)
    : ConstrainedScheduler(dataDepGraph, machineModel, upperBound) {
  VrfySched_ = vrfySched;
  prirts_ = priorities;
  rdyLst_ = new ReadyList(dataDepGraph_, priorities);
  count_ = dataDepGraph->GetInstCnt();
  Config &schedIni = SchedulerOptions::getInstance();

  use_fixed_bias = schedIni.GetBool("ACO_USE_FIXED_BIAS");
  heuristicImportance_ = schedIni.GetInt("ACO_HEURISTIC_IMPORTANCE");
  use_tournament = schedIni.GetBool("ACO_TOURNAMENT");
  fixed_bias = schedIni.GetInt("ACO_FIXED_BIAS");
  bias_ratio = schedIni.GetFloat("ACO_BIAS_RATIO");
  local_decay = schedIni.GetFloat("ACO_LOCAL_DECAY");
  decay_factor = schedIni.GetFloat("ACO_DECAY_FACTOR");
  ants_per_iteration = schedIni.GetInt("ACO_ANT_PER_ITERATION");
  print_aco_trace = schedIni.GetBool("ACO_TRACE");

  //pheremone Graph Debugging start
  std::string tgtRgns = schedIni.GetString("ACO_DBG_REGIONS");
  outPath = schedIni.GetString("ACO_DBG_REGIONS_OUT_PATH");
  if(tgtRgns!="NONE"){
    std::size_t startIdx = 0;
    std::size_t sepIdx = tgtRgns.find("|");
    while(sepIdx!=std::string::npos){
      dbgRgns.insert(tgtRgns.substr(startIdx,sepIdx-startIdx));
      startIdx = sepIdx+1;
      sepIdx = tgtRgns.find("|", startIdx);
    }
  }
  isDbg = dbgRgns.count(dataDepGraph_->GetDagID());
  //pheremone Graph Debugging end


  /*
  std::cerr << "useOldAlg===="<<useOldAlg<<"\n\n";
  std::cerr << "heuristicImportance_===="<<heuristicImportance_<<"\n\n";
  std::cerr << "tournament===="<<tournament<<"\n\n";
  std::cerr << "bias_ratio===="<<bias_ratio<<"\n\n";
  std::cerr << "local_decay===="<<local_decay<<"\n\n";
  std::cerr << "decay_factor===="<<decay_factor<<"\n\n";
  std::cerr << "ants_per_iteration===="<<ants_per_iteration<<"\n\n";
  */
  int pheremone_size = (count_ + 1) * count_;
  pheremone_.resize(pheremone_size);
  InitialSchedule = nullptr;
}

ACOScheduler::~ACOScheduler() { delete rdyLst_; }

// Pheremone table lookup
// -1 means no instruction, so e.g. pheremone(-1, 10) gives pheremone on path
// from empty schedule to schedule only containing instruction 10
pheremone_t &ACOScheduler::Pheremone(SchedInstruction *from,
                                     SchedInstruction *to) {
  assert(to != NULL);
  int fromNum = -1;
  if (from != NULL)
    fromNum = from->GetNum();
  return Pheremone(fromNum, to->GetNum());
}

pheremone_t &ACOScheduler::Pheremone(InstCount from, InstCount to) {
  int row = 0;
  if (from != -1)
    row = from + 1;
  return pheremone_[(row * count_) + to];
}

double ACOScheduler::Score(SchedInstruction *from, Choice choice) {
  return Pheremone(from, choice.inst) *
         pow(choice.heuristic, heuristicImportance_);
}

SchedInstruction *
ACOScheduler::SelectInstruction(const llvm::ArrayRef<Choice> &ready,
                                SchedInstruction *lastInst) {
#if USE_ACS
  double choose_best_chance;
  if (use_fixed_bias)
    choose_best_chance = fmax(0, 1 - (double)fixed_bias / count_);
  else
    choose_best_chance = bias_ratio;

  if (RandDouble(0, 1) < choose_best_chance) {
    if (print_aco_trace)
      std::cerr << "choose_best, use fixed bias: " << use_fixed_bias << "\n";
    pheremone_t max = -1;
    Choice maxChoice;
    for (auto &choice : ready) {
      if (Score(lastInst, choice) > max) {
        max = Score(lastInst, choice);
        maxChoice = choice;
      }
    }
    return maxChoice.inst;
  }
#endif
  if (use_tournament) {
    int POPULATION_SIZE = ready.size();
    int r_pos = (int)(RandDouble(0, 1) * POPULATION_SIZE);
    int s_pos = (int)(RandDouble(0, 1) * POPULATION_SIZE);
    //    int t_pos = (int) (RandDouble(0, 1) *POPULATION_SIZE);
    Choice r = ready[r_pos];
    Choice s = ready[s_pos];
    //    Choice t = ready[t_pos];
    if (print_aco_trace) {
      std::cerr << "tournament Start \n";
      std::cerr << "array_size:" << POPULATION_SIZE << "\n";
      std::cerr << "r:\t" << r_pos << "\n";
      std::cerr << "s:\t" << s_pos << "\n";
      //        std::cerr<<"t:\t"<<t_pos<<"\n";

      std::cerr << "Score r" << Score(lastInst, r) << "\n";
      std::cerr << "Score s" << Score(lastInst, s) << "\n";
      //         std::cerr<<"Score t"<<Score(lastInst, t)<<"\n";
    }
    if (Score(lastInst, r) >=
        Score(lastInst, s)) //&& Score(lastInst, r) >= Score(lastInst, t))
      return r.inst;
    //     else if (Score(lastInst, s) >= Score(lastInst, r) && Score(lastInst,
    //     s) >= Score(lastInst, t))
    //         return s.inst;
    else
      return s.inst;
  }
  pheremone_t sum = 0;
  for (auto choice : ready)
    sum += Score(lastInst, choice);
  pheremone_t point = RandDouble(0, sum);
  for (auto choice : ready) {
    point -= Score(lastInst, choice);
    if (point <= 0)
      return choice.inst;
  }
  std::cerr << "returning last instruction" << std::endl;
  assert(point < 0.001); // floats should not be this inaccurate
  return ready.back().inst;
}

std::unique_ptr<InstSchedule> ACOScheduler::FindOneSchedule() {
  SchedInstruction *lastInst = NULL;
  std::unique_ptr<InstSchedule> schedule =
      llvm::make_unique<InstSchedule>(machMdl_, dataDepGraph_, true);
  InstCount maxPriority = rdyLst_->MaxPriority();
  if (maxPriority == 0)
    maxPriority = 1; // divide by 0 is bad
  Initialize_();
  rgn_->InitForSchdulng();

  //graph debugging
  SmallVector<InstCount, 0> chosenPath;

  llvm::SmallVector<Choice, 0> ready;
  while (!IsSchedComplete_()) {
    // convert the ready list from a custom priority queue to a std::vector,
    // much nicer for this particular scheduler
    UpdtRdyLst_(crntCycleNum_, crntSlotNum_);
    unsigned long heuristic;
    ready.reserve(rdyLst_->GetInstCnt());
    SchedInstruction *inst = rdyLst_->GetNextPriorityInst(heuristic);
    while (inst != NULL) {
      if (ChkInstLglty_(inst)) {
        Choice c;
        c.inst = inst;
        c.heuristic = (double)heuristic / maxPriority;
        ready.push_back(c);
        if (isDbg&&lastInst)
          lastHeu[std::make_pair(lastInst->GetNum(),inst->GetNum())]=c.heuristic;
      }
      inst = rdyLst_->GetNextPriorityInst(heuristic);
    }
    rdyLst_->ResetIterator();

    // print out the ready list for debugging
    /*
     std::stringstream stream;
     stream << "Ready list: ";
    for (auto choice : ready) {
      stream << choice.inst->GetNum() << ", ";
    }
    Logger::Info(stream.str().c_str());
    */

    inst = NULL;
    if (!ready.empty())
      inst = SelectInstruction(ready, lastInst);
    if (inst != NULL) {
#ifdef USE_ACS
      // local pheremone decay
      pheremone_t *pheremone = &Pheremone(lastInst, inst);
      *pheremone = (1 - local_decay) * *pheremone + local_decay * initialValue_;
#endif
      if(isDbg&&lastInst!=NULL) {
        antEdges.insert(std::make_pair(lastInst->GetNum(),inst->GetNum()));
        crntAntEdges.insert(std::make_pair(lastInst->GetNum(),inst->GetNum()));
      }
      lastInst = inst;
    }

    // boilerplate, mostly copied from ListScheduler, try not to touch it
    InstCount instNum;
    if (inst == NULL) {
      instNum = SCHD_STALL;
    } else {
      instNum = inst->GetNum();
      SchdulInst_(inst, crntCycleNum_);
      inst->Schedule(crntCycleNum_, crntSlotNum_);
      rgn_->SchdulInst(inst, crntCycleNum_, crntSlotNum_, false);
      DoRsrvSlots_(inst);
      // this is annoying
      SchedInstruction *blah = rdyLst_->GetNextPriorityInst();
      while (blah != NULL && blah != inst) {
        blah = rdyLst_->GetNextPriorityInst();
      }
      if (blah == inst)
        rdyLst_->RemoveNextPriorityInst();
      UpdtSlotAvlblty_(inst);
    }
    /* Logger::Info("Chose instruction %d (for some reason)", instNum); */
    schedule->AppendInst(instNum);
    if (MovToNxtSlot_(inst))
      InitNewCycle_();
    rdyLst_->ResetIterator();
    ready.clear();
  }
  rgn_->UpdateScheduleCost(schedule.get());
  return schedule;
}

FUNC_RESULT ACOScheduler::FindSchedule(InstSchedule *schedule_out,
                                       SchedRegion *region) {
  rgn_ = region;

  // initialize pheremone
  // for this, we need the cost of the pure heuristic schedule
  int pheremone_size = (count_ + 1) * count_;
  for (int i = 0; i < pheremone_size; i++)
    pheremone_[i] = 1;
  initialValue_ = 1;
  std::unique_ptr<InstSchedule> heuristicSched = FindOneSchedule();
  InstCount heuristicCost =
      heuristicSched->GetCost() + 1; // prevent divide by zero

#if USE_ACS
  initialValue_ = 2.0 / ((double)count_ * heuristicCost);
#else
  initialValue_ = (double)ants_per_iteration / heuristicCost;
#endif
  for (int i = 0; i < pheremone_size; i++)
    pheremone_[i] = initialValue_;
  std::cerr << "initialValue_" << initialValue_ << std::endl;

  writePheremoneGraph("initial");

  std::unique_ptr<InstSchedule> bestSchedule = std::move(InitialSchedule);
  if (bestSchedule) {
    UpdatePheremone(bestSchedule.get());
  }
  Config &schedIni = SchedulerOptions::getInstance();
  int noImprovementMax = schedIni.GetInt("ACO_STOP_ITERATIONS");
  int noImprovement = 0; // how many iterations with no improvement
  int iterations = 0;
  while (true) {
    std::unique_ptr<InstSchedule> iterationBest;
    for (int i = 0; i < ants_per_iteration; i++) {
      crntAntEdges.clear();
      std::unique_ptr<InstSchedule> schedule = FindOneSchedule();
      if (print_aco_trace)
        PrintSchedule(schedule.get());
      if (iterationBest == nullptr ||
          schedule->GetCost() < iterationBest->GetCost()) {
        iterationBest = std::move(schedule);
        if(isDbg) iterAntEdges = crntAntEdges;
      }
    }
    UpdatePheremone(iterationBest.get());
    /* PrintSchedule(iterationBest); */
    /* std::cout << iterationBest->GetCost() << std::endl; */
    // TODO DRY
    if (bestSchedule == nullptr ||
        iterationBest->GetCost() < bestSchedule->GetCost()) {
      bestSchedule = std::move(iterationBest);
      Logger::Info("ACO found schedule with spill cost %d",
                   bestSchedule->GetCost());
      Logger::Info("ACO found schedule "
                   "cost:%d, rp cost:%d, sched length: %d, and "
                   "iteration:%d",
                   bestSchedule->GetCost(), bestSchedule->GetSpillCost(),
                   bestSchedule->GetCrntLngth(), iterations);
      if(isDbg) bestSched = iterAntEdges;

      noImprovement = 0;
    } else {
      noImprovement++;
      /* if (*iterationBest == *bestSchedule) */
      /*   std::cout << "same" << std::endl; */
      if (noImprovement > noImprovementMax)
        break;
    }

    writePheremoneGraph("iteration"+std::to_string(iterations));
    iterations++;
  }
  PrintSchedule(bestSchedule.get());
  schedule_out->Copy(bestSchedule.release());

  Logger::Info("ACO finished after %d iterations", iterations);
  return RES_SUCCESS;
}

void ACOScheduler::UpdatePheremone(InstSchedule *schedule) {
  // I wish InstSchedule allowed you to just iterate over it, but it's got this
  // cycle and slot thing which needs to be accounted for
  InstCount instNum, cycleNum, slotNum;
  instNum = schedule->GetFrstInst(cycleNum, slotNum);

  SchedInstruction *lastInst = NULL;
  while (instNum != INVALID_VALUE) {
    SchedInstruction *inst = dataDepGraph_->GetInstByIndx(instNum);

    pheremone_t *pheremone = &Pheremone(lastInst, inst);
#if USE_ACS
    // ACS update rule includes decay
    // only the arcs on the current solution are decayed
    *pheremone = (1 - decay_factor) * *pheremone +
                 decay_factor / (schedule->GetCost() + 1);
#else
    *pheremone = *pheremone + 1 / (schedule->GetCost() + 1);
#endif
    lastInst = inst;

    instNum = schedule->GetNxtInst(cycleNum, slotNum);
  }
  schedule->ResetInstIter();

#if !USE_ACS
  // decay pheremone
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      Pheremone(i, j) *= (1 - decay_factor);
    }
  }
#endif
  if (print_aco_trace)
    PrintPheremone();
}

// copied from Enumerator
inline void ACOScheduler::UpdtRdyLst_(InstCount cycleNum, int slotNum) {
  InstCount prevCycleNum = cycleNum - 1;
  LinkedList<SchedInstruction> *lst1 = NULL;
  LinkedList<SchedInstruction> *lst2 = frstRdyLstPerCycle_[cycleNum];

  if (slotNum == 0 && prevCycleNum >= 0) {
    // If at the begining of a new cycle other than the very first cycle, then
    // we also have to include the instructions that might have become ready in
    // the previous cycle due to a zero latency of the instruction scheduled in
    // the very last slot of that cycle [GOS 9.8.02].
    lst1 = frstRdyLstPerCycle_[prevCycleNum];

    if (lst1 != NULL) {
      rdyLst_->AddList(lst1);
      lst1->Reset();
      CleanupCycle_(prevCycleNum);
    }
  }

  if (lst2 != NULL) {
    rdyLst_->AddList(lst2);
    lst2->Reset();
  }
}

void ACOScheduler::PrintPheremone() {
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      std::cerr << std::scientific << std::setprecision(8) << Pheremone(i, j)
                << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
}

#ifndef NDEBUG
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static void PrintInstruction(SchedInstruction *inst) {
  std::cerr << std::setw(2) << inst->GetNum() << " ";
  std::cerr << std::setw(20) << std::left << inst->GetOpCode();

  std::cerr << " defs ";
  for (auto def : llvm::enumerate(inst->GetDefs())) {
    if (def.index() != 0)
      std::cerr << ", ";
    std::cerr << def.value()->GetNum() << def.value()->GetType();
  }

  std::cerr << " uses ";
  for (auto use : llvm::enumerate(inst->GetUses())) {
    if (use.index() != 0)
      std::cerr << ", ";
    std::cerr << use.value()->GetNum() << use.value()->GetType();
  }
  std::cerr << std::endl;
}
#endif

void PrintSchedule(InstSchedule *schedule) {
  std::cerr << schedule->GetCost() << ": ";
  InstCount instNum, cycleNum, slotNum;
  instNum = schedule->GetFrstInst(cycleNum, slotNum);
  while (instNum != INVALID_VALUE) {
    std::cerr << instNum << " ";
    instNum = schedule->GetNxtInst(cycleNum, slotNum);
  }
  std::cerr << std::endl;
  schedule->ResetInstIter();
}

void ACOScheduler::setInitialSched(InstSchedule *Sched) {
  if (Sched) {
    InitialSchedule =
        llvm::make_unique<InstSchedule>(machMdl_, dataDepGraph_, VrfySched_);
    InitialSchedule->Copy(Sched);
  }
}

void ACOScheduler::writePheremoneGraph(std::string stage) {
  if(!isDbg)
    return;

  std::string fullOutPath = outPath+"/"+dataDepGraph_->GetDagID()+"@"+stage+".dot";
  FILE* out = fopen(fullOutPath.c_str(), "w");
  if(!out){
    Logger::Info("Could now open file to write pheremone display at %s."
                 " Skipping.", fullOutPath.c_str());
    return;
  }

  //already added set
  llvm::SetVector<SchedInstruction*> visited;

  //header for .dot file
  fprintf(out, "digraph pheremone_matrix {\n");
  fprintf(out, "label=\"%s@%s\"\n", dataDepGraph_->GetDagID(), stage.c_str());
  fprintf(out, "ranksep=1.0\n");
/*
  for(InstCount i=-1; i<count_; ++i){
    for(InstCount j=0; j<count_; ++j){
      fprintf(out,"\t%d -> %d [label=\"%.4f\"];\n", i, j, Pheremone(i,j));
    }
  }
*/

  //find the recursive neighbors
  dataDepGraph_->FindRcrsvNghbrs(DIR_FRWRD);
  dataDepGraph_->FindRcrsvNghbrs(DIR_BKWRD);

  writePGraphRecursive(out, dataDepGraph_->GetRootInst(), visited);

  //footer for .dot file
  fprintf(out, "}\n");
  fclose(out);

  //wipe out antEdges
  antEdges.clear();
  crntAntEdges.clear();
  iterAntEdges.clear();
  lastHeu.clear();
}

std::string ACOScheduler::graphDisplayAnnotation(int frm, int to)
{
  std::string annotation;
  std::pair<InstCount, InstCount> thisEdge=std::make_pair(frm,to);

  if (bestSched.count(thisEdge)&&iterAntEdges.count(thisEdge))
    return "penwidth=2.0 color=\"#00FFFF\"";
  else if (iterAntEdges.count(thisEdge))
    return "penwidth=2.0 color=\"#00FF00\"";
  else if (bestSched.count(thisEdge))
    return "penwidth=2.0 color=\"#0000FF\"";
  else if (antEdges.count(thisEdge))
    return "color=\"#FF0000\"";
  else
    return "color=\"#000000\"";
}

std::string ACOScheduler::getHeuIfPossible(int frm, int to)
{
  std::pair<InstCount, InstCount> thisEdge=std::make_pair(frm,to);
  if(lastHeu.count(thisEdge)) {
    return std::string("|") + std::to_string(lastHeu[thisEdge]);
  }
  else return "";
}

void ACOScheduler::writePGraphRecursive(FILE* out, SchedInstruction* ins,
                                        llvm::SetVector<SchedInstruction*>& visited){
  InstCount i=ins->GetNum();

  //do not add edges out twice
  if(visited.count(ins)) return;
  //add self to set so edges are not double counted
  visited.insert(ins);

  //create edges for other orderings
  for(SchedInstruction* vIns : visited){
    if(!(ins->IsRcrsvPrdcsr(vIns)||ins->IsRcrsvScsr(vIns)))
    {
      InstCount vVtx=vIns->GetNum();
      std::string toAnno  = graphDisplayAnnotation(i,vVtx);
      std::string frmAnno = graphDisplayAnnotation(vVtx,i);
      std::string toHeu  = getHeuIfPossible(i,vVtx);
      std::string frmHeu = getHeuIfPossible(vVtx,i);

      fprintf(out, "\t%d -> %d [label=\"%f%s\" constraint=false style=dotted %s];\n", i, vVtx, Pheremone(i,vVtx), toHeu.c_str(),  toAnno.c_str());
      fprintf(out, "\t%d -> %d [label=\"%f%s\" constraint=false style=dotted %s];\n", vVtx, i, Pheremone(vVtx,i), frmHeu.c_str(), frmAnno.c_str());
    }
  }

  //add edges to children
  for(SchedInstruction* child = ins->GetFrstScsr(); child!=NULL; child= ins->GetNxtScsr()){
    InstCount j=child->GetNum();
    std::string sAnno = graphDisplayAnnotation(i,j);
    std::string sHeu  = getHeuIfPossible(i,j);
    fprintf(out, "\t%d -> %d [label=\"%f%s\" %s];\n", i, j, Pheremone(i,j), sHeu.c_str(), sAnno.c_str());
    writePGraphRecursive(out,child,visited);
  }
}
