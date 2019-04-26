import numpy as np
cimport numpy as np
from itertools import chain

ctypedef np.int_t DTYPE_t
DTYPE = np.int

cdef class SkipGram:
    cdef int data_index, group_index, epoch, batch_size, input_cache
    cdef DTYPE_t[:] context_cache

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

    def __call__(self, list data):
        cdef np.ndarray[DTYPE_t, ndim=1] batch  = np.ndarray(shape=(self.batch_size), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] labels = np.ndarray(shape=(self.batch_size, 1), dtype=DTYPE)
        if self.group_index >= len(data)-1:
            self.epoch += 1
            self.data_index = 0
            self.group_index = 0
        
        if self.data_index == len(data[self.group_index]):
            self.data_index = 0
            self.group_index = self.group_index + 1
        
        # locate group
        cdef np.ndarray[DTYPE_t, ndim=1] data_ = np.array(data[self.group_index])
        cdef int i = 0
        cdef DTYPE_t input_
        cdef list slides
        cdef np.ndarray[DTYPE_t, ndim=1] context_words

        while True:
            if self.input_cache < 0:
                input_ = data_[self.data_index]
                slides = list(chain(range(self.data_index), range(self.data_index+1, data_.shape[0])))
                if data_.shape[0]-1 > self.batch_size-i:
                    context_words = data_[slides[:self.batch_size-i]]
                    self.input_cache = input_
                    self.context_cache = data_[slides[self.batch_size-i:]]
                else:
                    context_words = data_[slides]
            else:
                input_ = self.input_cache
                context_words = np.asarray(self.context_cache)
                self.input_cache = -1
            for context_word in context_words:
                batch[i] = input_
                labels[i, 0] = context_word
                i += 1
                if i >= self.batch_size:
                    return batch, labels
            self.data_index += 1
            if self.data_index == data_.shape[0]:
                if self.group_index < len(data)-1: self.group_index += 1
                else:
                    self.group_index = 0
                    self.epoch += 1
                data_ = np.array(data[self.group_index])
                self.data_index = 0