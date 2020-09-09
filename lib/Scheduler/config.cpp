#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/logger.h"
#include "llvm/Support/ErrorHandling.h"
#include <fstream>
#include <sstream>

using namespace llvm::opt_sched;

using std::istringstream;

template <class T> T Convert(const string &value) {
  istringstream ss(value);
  T number = 0;
  ss >> number;
  assert(!ss.fail());
  return number;
}

template <class T> list<T> Split(const string &value) {
  list<T> values;
  if (value == "")
    return values;

  istringstream ss(value);
  while (ss) {
    T item;
    char delimiter;
    ss >> item;
    assert(!ss.fail());
    ss >> delimiter;
    assert(ss.fail() || delimiter == ',');
    values.push_back(item);
  }

  return values;
}

void Config::Load(const string &filepath) {
  std::ifstream file(filepath.c_str());
  Load(file);
}

void Config::Load(std::istream &file) {
  settings.clear();
  while (!file.eof()) {
    string name, value, comment;
    file >> name;
    while (!file.fail() && name.size() && name[0] == '#') {
      std::getline(file, comment);
      file >> name;
    }
    file >> value;
    while (!file.fail() && value.size() && value[0] == '#') {
      std::getline(file, comment);
      file >> value;
    }
    if (file.fail() || name == "" || value == "")
      break;
    settings[name] = value;
  }
}

string Config::GetString(const string &name) const {
  std::map<string, string>::const_iterator it = settings.find(name);
  if (it == settings.end()) {
    llvm::report_fatal_error("No value found for setting " + name, false);
    return "";
  } else {
    return it->second;
  }
}

string Config::GetString(const string &name, const string &default_) const {
  std::map<string, string>::const_iterator it = settings.find(name);
  if (it == settings.end()) {
    return default_;
  } else {
    return it->second;
  }
}

int64_t Config::GetInt(const string &name) const {
  return Convert<int64_t>(GetString(name));
}

int64_t Config::GetInt(const string &name, int64_t default_) const {
  if (settings.find(name) == settings.end()) {
    return default_;
  } else {
    return GetInt(name);
  }
}

float Config::GetFloat(const string &name) const {
  return Convert<float>(GetString(name));
}

float Config::GetFloat(const string &name, float default_) const {
  if (settings.find(name) == settings.end()) {
    return default_;
  } else {
    return GetFloat(name);
  }
}

bool Config::GetBool(const string &name) const {
  string value = GetString(name);
  if (value == "YES" || value == "yes" || value == "1" || value == "TRUE" ||
      value == "true") {
    return true;
  } else {
    assert(value == "NO" || value == "no" || value == "0" || value == "FALSE" ||
           value == "false");
    return false;
  }
}

bool Config::GetBool(const string &name, bool default_) const {
  if (settings.find(name) == settings.end()) {
    return default_;
  } else {
    return GetBool(name);
  }
}

list<string> Config::GetStringList(const string &name) const {
  list<string> values;
  string line = GetString(name, "");
  if (line == "")
    return values;

  istringstream ss(line);
  string item;

  while (std::getline(ss, item, ',')) {
    values.push_back(item);
  }

  return values;
}

list<int64_t> Config::GetIntList(const string &name) const {
  return Split<int64_t>(GetString(name, ""));
}

list<float> Config::GetFloatList(const string &name) const {
  return Split<float>(GetString(name, ""));
}

SchedulerOptions &SchedulerOptions::getInstance() {
  static SchedulerOptions instance; // The instance will always be destroyed.
  return instance;
}
