//(VLAD) Created my own class to replace SmallPtrSet in Register class
//since SmallPtrSet is not supported

template <typename T>
class DevicePtrSet {
  public:
    __device__
    DevicePtrSet(int size = 0) {
      size_ = size;
      alloc_ = size;

      if (alloc_ > 0)
        elmnt = new T[alloc_];
      else
        elmnt = NULL;
    }
    __device__
    ~DevicePtrSet() {
      if (elmnt)
        delete[] elmnt;
    }
    
    //inserts entry into set, returns true if
    //entry is a duplicate
    __device__
    bool insert(T entry) {
      bool dupl = false;
      if (alloc_ == 0) {
	alloc_ = 4;
        elmnt = new T[alloc_];
	elmnt[size_++] = entry;
      } else {
        //check if entry has already been entered
        for (int i = 0; i < size_; i++) {
          if (entry == elmnt[i])
            dupl = true;
        }
	//if not duplicate, insert
	if (!dupl) {
	  //allocate more space if full
	  if (alloc_ == size_) {
	    alloc_ *= 2;
	    T *new_arr = new T[alloc_];
	    //copy old array
	    for (int i = 0; i < size_; i++)
	      new_arr[i] = elmnt[i];
	    delete[] elmnt;
	    elmnt = new_arr;
	  }
	  //add entry to array
          elmnt[size_++] = entry;
	}
      }
      return dupl;
    }

    __device__
    int size() const {
      return size_;
    }

    //searches set for entry, returns true if match is found
    __device__
    bool contains(T entry) const {
      for (int i = 0; i < size_; i++) {
        if (elmnt[i] == entry)
          return true;
      }
      return false;
    }

    __device__
    T& operator[](int indx) {
      if (indx < size_ && indx >= 0)
        return elmnt[indx];
      else {
        printf("Index out of bounds!\n");
	return NULL;
      }
    }

    // Iterator Class
    class iterator {
    private:
        // Dynamic array using pointers
        T *ptr;

    public:
        explicit iterator()
            : ptr(nullptr)
        {
        }
        explicit iterator(T *p)
            : ptr(p)
        {
        }
        bool operator==(const iterator& rhs) const
        {
            return ptr == rhs.ptr;
        }
        bool operator!=(const iterator& rhs) const
        {
            return !(*this == rhs);
        }
        T operator*() const
        {
            return *ptr;
        }
        iterator& operator++()
        {
            ++ptr;
            return *this;
        }
        iterator operator++(int)
        {
            iterator temp(*this);
            ++*this;
            return temp;
        }
    };

    // Begin iterator
    iterator begin() const {
      return iterator(elmnt);
    }

    // End iterator
    iterator end() const {
      return iterator(elmnt + size_);
    }

  private:
    int size_;
    int alloc_;
    T *elmnt;
};
