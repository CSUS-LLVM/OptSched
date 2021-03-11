// (Vlad) Created new ArrayList class to replace LinkedList in SchedInstruction
// and GraphNode classes. Allows for easier copy to device but is slower with
// sorting, inserting (when not inserting at the end), and deleting since
// whole array must be copied in those cases. This class works on both host
// and device and contains methods to copy to device

#define END -1
#define GLOBALTID blockIdx.x * blockDim.x + threadIdx.x 

// Base ArrayList class, replaces LinkedList
template <typename T>
class ArrayList {
  public:
    __host__ __device__
    ArrayList(int maxSize = 0);
    
    __host__ __device__
    ~ArrayList();
    
    // Appends a new element to the end of the list
    __host__ __device__
    void InsrtElmnt(T elmnt);

    // Returns number of elements in ArrayList
    __host__ __device__
    int GetElmntCnt();

    // Returns first element, resets iterator
    __host__ __device__
    T GetFrstElmnt();

    // Returns next element in the array
    __host__ __device__
    T GetNxtElmnt();

    // returns next element allong with its index
    __host__ __device__
    T GetNxtElmnt(int &indx);

    // Returns previous element in the array
    __host__ __device__
    T GetPrevElmnt();

    // Returns last element in array
    __host__ __device__
    T GetLastElmnt();

    // Removes last element in array
    __host__ __device__
    void RmvLastElmnt();

    // Resets ArrayList to empty state
    __host__ __device__
    void Reset();

    // Resets iterator
    __host__ __device__
    void ResetIterator();

    // Removes elmnt at crnt_ index
    __host__ __device__
    void RmvCrntElmnt();

    // Calls FindElmnt with initialized hitCnt
    __host__ __device__
    bool FindElmnt(const T element) const;

    // Returns true if elmnt is found, places num of matches in
    // hitCnt. 
    __host__ __device__
    bool FindElmnt(const T element, int &hitCnt) const;

    // Removes elmnt that matches passed elmnt
    __host__ __device__
    void RmvElmnt(T elmnt);

    int maxSize_;
    int size_;
    int crnt_;
    int *dev_crnt_;
    T *elmnts_;
};

// Sorts its list based on provided key, replaces PriorityList
template <typename T, typename K = unsigned long>
class PriorityArrayList : public ArrayList<T> {
  public:
    __host__ __device__
    PriorityArrayList(int maxSize = 0);

    __host__ __device__
    ~PriorityArrayList();

    // Insert a new element by automatically finding its place in the list.
    // If allowDplct is false, the element will not be inserted if another
    // element with the same key exists.
    __host__ __device__
    void InsrtElmnt(T elmnt, K key, bool allowDplct);

    // Copy passed list to this
    __host__ __device__
    void CopyList(PriorityArrayList<T,K> *otherLst);

    // Remove elmnt and key at crnt_ indx
    __host__ __device__
    void RmvCrntElmnt();

    // Get Key at current index
    __host__ __device__
    K GetCrntKey() {
      return keys_[ArrayList<T>::crnt_];
    }

    // Returns key at passed index
    __host__ __device__
    K GetKey(int indx) {
      return keys_[indx];
    }

    // updates key and then moves elmnt if needed
    __host__ __device__
    void BoostElmnt(T elmnt, K newKey);
  
    K *keys_;

};

template <>
class ArrayList<int> {
  public:
    __host__ __device__
    ArrayList(int maxSize = 0) {
      maxSize_ = maxSize;
      size_ = 0;
      crnt_ = 0;
      dev_crnt_ = NULL;
      if (maxSize > 0)
        elmnts_ = new int[maxSize_];
      else
        elmnts_ = NULL;
    }

    __host__ __device__
    ~ArrayList() {
      if (elmnts_)
        delete[] elmnts_;
    }

    // Appends a new element to the end of the list
    __host__ __device__
    void InsrtElmnt(int elmnt) {
      if (size_ < maxSize_)
        elmnts_[size_++] = elmnt;
    }

    // Returns number of elements in ArrayList
    __host__ __device__
    int GetElmntCnt() {
      return size_;
    }

    // Returns first element, resets iterator
    __host__ __device__
    int GetFrstElmnt() {
#ifdef __CUDA_ARCH__
      if (dev_crnt_) {
        dev_crnt_[GLOBALTID] = 0;

        if (dev_crnt_[GLOBALTID] < size_)
          return elmnts_[dev_crnt_[GLOBALTID]];
        else
          return END;
      } else {
        crnt_ = 0;
  
        if (crnt_ < size_)
          return elmnts_[crnt_];
        else
          return END;
      }
#else
      crnt_ = 0;
  
      if (crnt_ < size_)
        return elmnts_[crnt_];
      else
        return END;
#endif
    }

    // Returns next element in the array
    __host__ __device__
    int GetNxtElmnt() {
#ifdef __CUDA_ARCH__
      if (dev_crnt_) {
        if (dev_crnt_[GLOBALTID] != size_)
          dev_crnt_[GLOBALTID]++;
        else
          return END;

        if (dev_crnt_[GLOBALTID] < size_ && dev_crnt_[GLOBALTID] >= 0)
          return elmnts_[dev_crnt_[GLOBALTID]];
        else
          return END;
      } else {
        if (crnt_ != size_)
          crnt_++;
        else
          return END;

        if (crnt_ < size_ && crnt_ >= 0)
          return elmnts_[crnt_];
        else
          return END;
      }
#else
      if (crnt_ != size_)
        crnt_++;
      else
        return END;

      if (crnt_ < size_ && crnt_ >= 0)
        return elmnts_[crnt_];
      else
        return END;
#endif
    }

    __host__ __device__
    int GetNxtElmnt(int &indx) {
#ifdef __CUDA_ARCH__
      if (dev_crnt_) {
        if (dev_crnt_[GLOBALTID] != size_)
          dev_crnt_[GLOBALTID]++;
        else
          return END;

        indx = dev_crnt_[GLOBALTID];

        if (dev_crnt_[GLOBALTID] < size_ && dev_crnt_[GLOBALTID] >= 0)
          return elmnts_[dev_crnt_[GLOBALTID]];
        else
          return END;
      } else {
        if (crnt_ != size_)
          crnt_++;
        else
          return END;

        indx = crnt_;

        if (crnt_ < size_ && crnt_ >= 0)
          return elmnts_[crnt_];
        else
          return END;
      }
#else
      if (crnt_ != size_)
        crnt_++;
      else
        return END;

      indx = crnt_;

      if (crnt_ < size_ && crnt_ >= 0)
        return elmnts_[crnt_];
      else
        return END;
#endif
    }

    // Returns previous element in the array
    __host__ __device__
    int GetPrevElmnt() {
#ifdef __CUDA_ARCH__
      if (dev_crnt_) {
        if (dev_crnt_[GLOBALTID] != 0)
          dev_crnt_[GLOBALTID]--;
        else
          return END;

        if (dev_crnt_[GLOBALTID] >= 0 && dev_crnt_[GLOBALTID] < size_)
          return elmnts_[dev_crnt_[GLOBALTID]];
        else
          return END;
      } else {
        if (crnt_ != 0)
          crnt_--;
        else
          return END;

        if (crnt_ >= 0 && crnt_ < size_)
          return elmnts_[crnt_];
        else
          return END;
      }
#else
      if (crnt_ != 0)
        crnt_--;
      else
        return END;

      if (crnt_ >= 0 && crnt_ < size_)
        return elmnts_[crnt_];
      else
        return END;
#endif
    }

    // Returns last element in array
    __host__ __device__
    int GetLastElmnt() {
#ifdef __CUDA_ARCH__
      if (dev_crnt_) {
        if (size_ > 0)
          dev_crnt_[GLOBALTID] = size_ - 1;
        else
          return END;

        return elmnts_[dev_crnt_[GLOBALTID]];
      } else {
        if (size_ > 0)
          crnt_ = size_ - 1;
        else
          return END;

        return elmnts_[crnt_];
      }
#else
      if (size_ > 0)
        crnt_ = size_ - 1;
      else 
	return END;

      return elmnts_[crnt_];
#endif
    }

    // Removes last element in array
    __host__ __device__
    void RmvLastElmnt() {
      size_--;
    }

    // Resets ArrayList to empty state
    __host__ __device__
    void Reset() {
      size_ = 0;
    }

    __host__ __device__
    void ResetIterator() {
#ifdef __CUDA_ARCH__
      if (dev_crnt_)
        dev_crnt_[GLOBALTID] = -1;
      else
        crnt_ = -1;
#else
      crnt_ = -1;
#endif
    }

    __host__ __device__
    bool FindElmnt(const int element, int &hitCnt) const {
      for (int i = 0; i < size_; i++) {
        if (element == elmnts_[i])
          hitCnt++;
        }
      return hitCnt > 0 ? true : false;
    }

    __host__ __device__
    bool FindElmnt(const int element) const {
      int hitCnt;
      return FindElmnt(element, hitCnt);
    }

    __host__ __device__
    void RmvElmnt(int elmnt) {
      int elmntIndx;
      for (int i = 0; i < size_; i++) {
        if (elmnts_[i] == elmnt) {
          elmntIndx = i;
          break;
        }
      }
      size_--;
      for (int i = elmntIndx; i < size_; i++)
        elmnts_[i] = elmnts_[i + 1];
    }

    int maxSize_;
    int size_;
    int crnt_;
    int *dev_crnt_;
    int *elmnts_;
};

template <typename T>
__host__ __device__
ArrayList<T>::ArrayList(int maxSize) {
  maxSize_ = maxSize;
  size_ = 0;
  crnt_ = 0;
  dev_crnt_ = NULL;
  if (maxSize > 0)
    elmnts_ = new T[maxSize_];
  else
    elmnts_ = NULL;
}

template <typename T>
__host__ __device__
ArrayList<T>::~ArrayList() {
  if (elmnts_)
    delete[] elmnts_;
}

template <typename T>
__host__ __device__
void ArrayList<T>::InsrtElmnt(T elmnt) {
  if (size_ < maxSize_)
    elmnts_[size_++] = elmnt;
}

template <typename T>
__host__ __device__
int ArrayList<T>::GetElmntCnt() {
  return size_;
}

template <typename T>
__host__ __device__
T ArrayList<T>::GetFrstElmnt() {
#ifdef __CUDA_ARCH__
  if (dev_crnt_) {
    dev_crnt_[GLOBALTID] = 0;

    if (dev_crnt_[GLOBALTID] < size_)
      return elmnts_[dev_crnt_[GLOBALTID]];
    else
      return NULL;
  } else {
    crnt_ = 0;

    if (crnt_ < size_)
      return elmnts_[crnt_];
    else
      return NULL;
  }
#else
  crnt_ = 0;
  
  if (crnt_ < size_)
    return elmnts_[crnt_];
  else
    return NULL;
#endif
}

template <typename T>
__host__ __device__
T ArrayList<T>::GetNxtElmnt() {
#ifdef __CUDA_ARCH__
  if (dev_crnt_) {
    if (dev_crnt_[GLOBALTID] != size_)
      dev_crnt_[GLOBALTID]++;
    else
      return NULL;

    if (dev_crnt_[GLOBALTID] < size_ && dev_crnt_[GLOBALTID] >= 0)
      return elmnts_[dev_crnt_[GLOBALTID]];
    else
      return NULL;
  } else {
    if (crnt_ != size_)
      crnt_++;
    else
      return NULL;

    if (crnt_ < size_ && crnt_ >= 0)
      return elmnts_[crnt_];
    else
      return NULL;
  }
#else
  if (crnt_ != size_)
    crnt_++;
  else
    return NULL;

  if (crnt_ < size_ && crnt_ >= 0)
    return elmnts_[crnt_];
  else
    return NULL;
#endif
}

template <typename T>
__host__ __device__
T ArrayList<T>::GetNxtElmnt(int &indx) {
#ifdef __CUDA_ARCH__
  if (dev_crnt_) {
    if (dev_crnt_[GLOBALTID] != size_)
      dev_crnt_[GLOBALTID]++;
    else
      return NULL;

    indx = dev_crnt_[GLOBALTID];

    if (dev_crnt_[GLOBALTID] < size_ && dev_crnt_[GLOBALTID] >= 0)
      return elmnts_[dev_crnt_[GLOBALTID]];
    else
      return NULL;
  } else {
    if (crnt_ != size_)
      crnt_++;
    else
      return NULL;

    indx = crnt_;

    if (crnt_ < size_ && crnt_ >= 0)
      return elmnts_[crnt_];
    else
      return NULL;
  }
#else
  if (crnt_ != size_)
    crnt_++;
  else
    return NULL;

  indx = crnt_;

  if (crnt_ < size_ && crnt_ >= 0)
    return elmnts_[crnt_];
  else
    return NULL;
#endif
}

template <typename T>
__host__ __device__
T ArrayList<T>::GetPrevElmnt() {
#ifdef __CUDA_ARCH__
  if (dev_crnt_) {
    if (dev_crnt_[GLOBALTID] != 0)
      dev_crnt_[GLOBALTID]--;
    else
      return NULL;

    if (dev_crnt_[GLOBALTID] >= 0 && dev_crnt_[GLOBALTID] < size_)
      return elmnts_[dev_crnt_[GLOBALTID]];
    else
      return NULL;
  } else {
    if (crnt_ != 0)
      crnt_--;
    else
      return NULL;

    if (crnt_ >= 0 && crnt_ < size_)
      return elmnts_[crnt_];
    else
      return NULL;
  }
#else
  if (crnt_ != 0)
    crnt_--;
  else
    return NULL;

  if (crnt_ >= 0 && crnt_ < size_)
    return elmnts_[crnt_];
  else
    return NULL;
#endif
}

template <typename T>
__host__ __device__
T ArrayList<T>::GetLastElmnt() {
#ifdef __CUDA_ARCH__
  if (dev_crnt_) {
    if (size_ > 0)
      dev_crnt_[GLOBALTID] = size_ - 1;
    else
      return NULL;

    return elmnts_[dev_crnt_[GLOBALTID]];
  } else {
    if (size_ > 0)
      crnt_ = size_ - 1;
    else
      return NULL;

    return elmnts_[crnt_];
  }
#else
  if (size_ > 0)
    crnt_ = size_ - 1;
  else
    return NULL;
  
  return elmnts_[crnt_];
#endif
}

template <typename T>
__host__ __device__
void ArrayList<T>::RmvLastElmnt() {
  if (size_ > 0)
    size_--;
}

template <typename T>
__host__ __device__
void ArrayList<T>::Reset() {
  size_ = 0;
}

template <typename T>
__host__ __device__
void ArrayList<T>::ResetIterator() {
#ifdef __CUDA_ARCH__
  if (dev_crnt_)
    dev_crnt_[GLOBALTID] = -1;
  else
    crnt_ = -1;
#else
  crnt_ = -1;
#endif
}

template <typename T>
__host__ __device__
void ArrayList<T>::RmvCrntElmnt() {
  assert(crnt_ != -1);
  assert(size_ > 0);
  size_--;
  for (int i = crnt_; i < size_; i++)
    elmnts_[i] = elmnts_[i + 1];
}

template <typename T>
__host__ __device__
bool ArrayList<T>::FindElmnt(const T element, int &hitCnt) const {
  for (int i = 0; i < size_; i++) {
    if (element == elmnts_[i])
      hitCnt++;
  }
  return hitCnt > 0 ? true : false;
}

template <typename T>
__host__ __device__
void ArrayList<T>::RmvElmnt(T elmnt) {
  int elmntIndx;
  for (int i = 0; i < size_; i++) {
    if (elmnts_[i] == elmnt) {
      elmntIndx = i;
      break;
    }
  }
  size_--;
  for (int i = elmntIndx; i < size_; i++)
    elmnts_[i] = elmnts_[i + 1];
}

template <typename T>
__host__ __device__
bool ArrayList<T>::FindElmnt(const T element) const {
  int hitCnt;
  return FindElmnt(element, hitCnt);
}

template <typename T, typename K>
__host__ __device__
PriorityArrayList<T,K>::PriorityArrayList(int maxSize) : ArrayList<T>(maxSize) {
  if (maxSize > 0)
    keys_ = new K[maxSize];
  else
    keys_ = NULL;
}

template <typename T, typename K>
__host__ __device__
PriorityArrayList<T,K>::~PriorityArrayList() {
  if (keys_)
    delete[] keys_;
}

template <typename T, typename K>
__host__ __device__
void PriorityArrayList<T,K>::InsrtElmnt(T elmnt, K key, bool allowDplct) {
  // Array is full
  if (ArrayList<T>::size_ == ArrayList<T>::maxSize_)
    return;
	
  if (ArrayList<T>::size_ == 0) {
    ArrayList<T>::elmnts_[ArrayList<T>::size_] = elmnt;
    keys_[ArrayList<T>::size_] = key;
    ArrayList<T>::size_++;
    return;
  }
  if (allowDplct) { // Do reverse insertion
    for (int i = ArrayList<T>::size_ - 1; i > -2; i--) {
      if (i == -1 || keys_[i] >= key) {
        ArrayList<T>::elmnts_[i + 1] = elmnt;
        keys_[i + 1] = key;
        ArrayList<T>::size_++;
        break;
      }
      ArrayList<T>::elmnts_[i + 1] = ArrayList<T>::elmnts_[i];
      keys_[i + 1] = keys_[i];
    }
  } else {  // Do regular insert so we can scan for duplicates before shifting
    int indx;
    bool foundDplct = false;

    for (indx = 0; indx < ArrayList<T>::size_; indx++) {
      if (keys_[indx] <= key) {
        foundDplct = (keys_[indx] == key);
        break;
      }
    }

    if (!allowDplct && foundDplct)
      return;

    // if indx != size_ we must move all entries at and after indx to make
    // space for new elmnt
    if (indx != ArrayList<T>::size_) {
      for (int i = ArrayList<T>::size_; i > indx; i--) {
        ArrayList<T>::elmnts_[i] = ArrayList<T>::elmnts_[i - 1];
        keys_[i] = keys_[i - 1];
      }
    }

    ArrayList<T>::elmnts_[indx] = elmnt;
    keys_[indx] = key;
    ArrayList<T>::size_++;
  }
}

template <typename T, typename K>
__host__ __device__
void PriorityArrayList<T,K>::CopyList(PriorityArrayList<T,K> *otherLst) {
  for (int i = 0; i < otherLst->size_; i++) {
    InsrtElmnt(otherLst->elmnts_[i], otherLst->keys_[i], true);
  }
}

template <typename T, typename K>
__host__ __device__
void PriorityArrayList<T,K>::RmvCrntElmnt() {
  assert(ArrayList<T>::crnt_ != -1);
  assert(ArrayList<T>::size_ > 0);
  ArrayList<T>::size_--;
  for (int i = ArrayList<T>::crnt_; i < ArrayList<T>::size_; i++) {
    ArrayList<T>::elmnts_[i] = ArrayList<T>::elmnts_[i + 1];
    keys_[i] = keys_[i + 1];
  }
}

template <typename T, typename K>
__host__ __device__
void PriorityArrayList<T,K>::BoostElmnt(T elmnt, K newKey) {
  int elmntIndx = -1;
  int newIndx;
  // FindElmnt
  for (int i = 0; i < ArrayList<T>::size_; i++) {
    if (elmnt == ArrayList<T>::elmnts_[i]) {
      elmntIndx = i;
      break;
    }
  }

  if (elmntIndx != -1) {
    if (keys_[elmntIndx] < newKey) {
      // if elmnt is already at the top or its prev elmnt still has a 
      // higher key, it is already in place
      if (elmntIndx == 0 || keys_[elmntIndx - 1] >= newKey)
        return;

      newIndx = elmntIndx;

      while (newIndx != 0 && keys_[newIndx - 1] < newKey)
        newIndx--;

      for (int i = elmntIndx; i > newIndx; i--) {
        ArrayList<T>::elmnts_[i] = ArrayList<T>::elmnts_[i - 1];
        keys_[i] = keys_[i - 1];
      }

      ArrayList<T>::elmnts_[newIndx] = elmnt;
      keys_[newIndx] = newKey;
    } else if (keys_[elmntIndx] < newKey) {
      // if elmnt is already at the bottom or next elmnt still has
      // a lower key, it is already in place
      if (elmntIndx == ArrayList<T>::size_ - 1 || 
	  keys_[elmntIndx + 1] <= newKey)
        return;

      newIndx = elmntIndx;

      while (newIndx != ArrayList<T>::size_ - 1 && 
	     keys_[newIndx + 1] > newKey)
        newIndx++;

      for (int i = elmntIndx; i < newIndx; i++) {
        ArrayList<T>::elmnts_[i] = ArrayList<T>::elmnts_[i + 1];
        keys_[i] = keys_[i + 1];
      }

      ArrayList<T>::elmnts_[newIndx] = elmnt;
      keys_[newIndx] = newKey;
    }
  }
}
