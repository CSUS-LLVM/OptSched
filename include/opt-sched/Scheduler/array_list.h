// (Vlad) Created new ArrayList class to replace LinkedList in SchedInstruction
// and GraphNode classes. Allows for easier copy to device but is slower with
// sorting, inserting (when not inserting at the end), and deleting since
// whole array must be copied in those cases. This class works on both host
// and device and contains methods to copy to device

#define END -1

// Base ArrayList class, replaces LinkedList
template <typename T>
class ArrayList {
  public:
    __host__ __device__
    ArrayList(int maxSize);
    
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

    int maxSize_;
    int size_;
    int crnt_;
    T *elmnts_;
};

// Sorts its list based on provided key, replaces PriorityList
template <typename T, typename K = unsigned long>
class PriorityArrayList : public ArrayList<T> {
  public:
    __host__ __device__
    PriorityArrayList(int maxSize);

    __host__ __device__
    ~PriorityArrayList();

    // Insert a new element by automatically finding its place in the list.
    // If allowDplct is false, the element will not be inserted if another
    // element with the same key exists.
    __host__ __device__
    void InsrtElmnt(T elmnt, K key, bool allowDplct);
  
    K *keys_;

};

template <>
class ArrayList<int> {
  public:
    __host__ __device__
    ArrayList(int maxSize) {
      maxSize_ = maxSize;
      size_ = 0;
      crnt_ = 0;
      elmnts_ = new int[maxSize_];
    }

    __host__ __device__
    ~ArrayList() {
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
      crnt_ = 0;
  
      if (crnt_ < size_)
        return elmnts_[crnt_];
      else
        return END;
    }

    // Returns next element in the array
    __host__ __device__
    int GetNxtElmnt() {
      if (crnt_ != size_)
        crnt_++;
      else
        return END;

      if (crnt_ < size_ && crnt_ >= 0)
        return elmnts_[crnt_];
      else
        return END;
    }

    // Returns previous element in the array
    __host__ __device__
    int GetPrevElmnt() {
      if (crnt_ != 0)
        crnt_--;
      else
        return END;

      if (crnt_ >= 0 && crnt_ < size_)
        return elmnts_[crnt_];
      else
        return END;
    }

    // Returns last element in array
    __host__ __device__
    int GetLastElmnt() {
      if (size_ > 0)
        crnt_ = size_ - 1;
      else 
	return END;

      return elmnts_[crnt_];
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

    int maxSize_;
    int size_;
    int crnt_;
    int *elmnts_;
};

template <typename T>
__host__ __device__
ArrayList<T>::ArrayList(int maxSize) {
  maxSize_ = maxSize;
  size_ = 0;
  crnt_ = 0;
  elmnts_ = new T[maxSize_];
}

template <typename T>
__host__ __device__
ArrayList<T>::~ArrayList() {
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
  crnt_ = 0;
  
  if (crnt_ < size_)
    return elmnts_[crnt_];
  else
    return NULL;
}

template <typename T>
__host__ __device__
T ArrayList<T>::GetNxtElmnt() {
  if (crnt_ != size_)
    crnt_++;
  else
    return NULL;

  if (crnt_ < size_ && crnt_ >= 0)
    return elmnts_[crnt_];
  else
    return NULL;
}

template <typename T>
__host__ __device__
T ArrayList<T>::GetPrevElmnt() {
  if (crnt_ != 0)
    crnt_--;
  else
    return NULL;

  if (crnt_ >= 0 && crnt_ < size_)
    return elmnts_[crnt_];
  else
    return NULL;
}

template <typename T>
__host__ __device__
T ArrayList<T>::GetLastElmnt() {
  if (size_ > 0)
    crnt_ = size_ - 1;
  else
    return NULL;
  
  return elmnts_[crnt_];
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

template <typename T, typename K>
__host__ __device__
PriorityArrayList<T,K>::PriorityArrayList(int maxSize) : ArrayList<T>(maxSize) {
  keys_ = new K[maxSize];
}

template <typename T, typename K>
__host__ __device__
PriorityArrayList<T,K>::~PriorityArrayList() {
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
