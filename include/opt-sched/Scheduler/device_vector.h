// (Vlad) Created my own vector class that can be used on host & device.

template <typename T>
class DeviceVector {
public:
  __host__ __device__
  DeviceVector(int size = 0);
 
  __host__ __device__
  ~DeviceVector();

  __host__ __device__
  void reserve(int reserveSize);

  __host__ __device__
  void resize(int newSize);

  __host__ __device__
  void push_back(T elmnt);

  __host__ __device__
  bool empty();

  __host__ __device__
  T back();

  __host__ __device__
  void clear();

  __host__ __device__
  void deallocate();

  __host__ __device__
  int size();

  __host__ __device__
  T& operator[](int indx);

  // Iterator Class
  class iterator {
    private:
        // Dynamic array using pointers
        T *ptr;

    public:
	__host__ __device__
        explicit iterator()
            : ptr(nullptr)
        {
        }
	__host__ __device__
        explicit iterator(T *p)
            : ptr(p)
        {
        }
        __host__ __device__
        bool operator==(const iterator& rhs) const
        {
            return ptr == rhs.ptr;
        }
        __host__ __device__
        bool operator!=(const iterator& rhs) const
        {
            return !(*this == rhs);
        }
        __host__ __device__
        T operator*() const
        {
            return *ptr;
        }
        __host__ __device__
        iterator& operator++()
        {
            ++ptr;
            return *this;
        }
        __host__ __device__
        iterator operator++(int)
        {
            iterator temp(*this);
            ++*this;
            return temp;
        }
    };

    // Begin iterator
    __host__ __device__
    iterator begin() const {
      return iterator(elmnts_);
    }

    // End iterator
    __host__ __device__
    iterator end() const {
      return iterator(elmnts_ + size_);
    }

  int size_;
  int alloc_;
  T *elmnts_;
};

template <typename T>
__host__ __device__
DeviceVector<T>::DeviceVector(int size) {
  size_ = 0;
  alloc_ = size;
  if (alloc_ > 0)
    elmnts_ = new T[alloc_];
  else
    elmnts_ = NULL;
}

template <typename T>
__host__ __device__
DeviceVector<T>::~DeviceVector() {
  if(elmnts_)
    delete[] elmnts_;
}

template <typename T>
__host__ __device__
void DeviceVector<T>::reserve(int reserveSize) {
  if (alloc_ < reserveSize) {
    T *new_elmnts = new T[reserveSize];
    if (size_ > 0) {
      for (int i = 0; i < size_; i++)
        new_elmnts[i] = elmnts_[i];
    }
    if (alloc_ > 0)
      delete[] elmnts_;
    alloc_ = reserveSize;
    elmnts_ = new_elmnts;
  }
}

template <typename T>
__host__ __device__
void DeviceVector<T>::resize(int newSize) {
  if (newSize < size_)
    size_ = newSize;
  
  reserve(newSize);
}

template <typename T>
__host__ __device__
void DeviceVector<T>::push_back(T elmnt) {
  if (alloc_ == 0)
    reserve(4);
  if (size_ == alloc_)
    reserve(alloc_ * 2);

  elmnts_[size_++] = elmnt;
}

template <typename T>
__host__ __device__
bool DeviceVector<T>::empty() {
  return size_ == 0;
}

template <typename T>
__host__ __device__
T DeviceVector<T>::back() {
  return elmnts_[size_ - 1];
}

template <typename T>
__host__ __device__
void DeviceVector<T>::clear() {
  size_ = 0;
}

template <typename T>
__host__ __device__
void DeviceVector<T>::deallocate() {
  delete[] elmnts_;
}

template <typename T>
__host__ __device__
int DeviceVector<T>::size() {
  return size_;
}

template <typename T>
__host__ __device__
T& DeviceVector<T>::operator[](int indx) {
  return elmnts_[indx];
}
