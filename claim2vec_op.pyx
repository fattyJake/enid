import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int_t DTYPE_t
DTYPE = np.int

cdef class SkipGram:
    cdef int data_index, group_index, epoch, batch_size, input_cache
    cdef list context_cache

    def __init__(self, int batch_size):
        self.data_index = 0
        self.group_index = 0
        self.epoch = 0
        self.batch_size = batch_size
        self.input_cache = -1

    def get_data_index(self):
        return self.data_index

    def get_group_index(self):
        return self.group_index

    def get_epoch(self):
        return self.epoch

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self, list data):
        cdef np.ndarray[DTYPE_t, ndim=1, mode="c"] batch  = np.ndarray(shape=(self.batch_size), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] labels = np.ndarray(shape=(self.batch_size, 1), dtype=DTYPE)
        if self.group_index >= len(data)-1:
            self.epoch += 1
            self.data_index = 0
            self.group_index = 0
        
        if self.data_index == len(data[self.group_index]):
            self.data_index = 0
            self.group_index = self.group_index + 1
        
        # locate group
        data_ = data[self.group_index]
        cdef int i = 0
        cdef int input_
        cdef list context_words

        while True:
            if self.input_cache < 0:
                input_ = data_[self.data_index]
                if len(data_)-1 > self.batch_size-i:
                    context_words = [data_[idx] for idx in range(self.batch_size-i) if idx != self.data_index]
                    self.input_cache = input_
                    self.context_cache = [data_[idx] for idx in range(self.batch_size-i, len(data_)) if idx != self.data_index]
                else:
                    context_words = [data_[idx] for idx in range(len(data_)) if idx != self.data_index]
            else:
                input_ = self.input_cache
                if len(self.context_cache) > self.batch_size-i:
                    context_words = self.context_cache[:self.batch_size-i]
                    self.context_cache = self.context_cache[self.batch_size-i:]
                else:
                    context_words = self.context_cache
                    self.input_cache = -1
                    self.context_cache = []
            for context_word in context_words:
                batch[i] = input_
                labels[i, 0] = context_word
                i += 1
                if i >= self.batch_size:
                    return batch, labels

            if self.input_cache < 0:
                self.data_index += 1
                if self.data_index == len(data_):
                    if self.group_index < len(data)-1: self.group_index += 1
                    else:
                        self.group_index = 0
                        self.epoch += 1
                    data_ = data[self.group_index]
                    self.data_index = 0