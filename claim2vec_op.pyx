import numpy as np
cimport numpy as np
import random

DTYPE = np.int

cdef class SkipGram:
    cdef int data_index, group_index, epoch, batch_size

    def __init__(self, int batch_size):
        self.data_index = 0
        self.group_index = 0
        self.epoch = 0
        self.batch_size = batch_size

    def __call__(self, list data):
        batch = np.ndarray(shape=(self.batch_size), dtype=DTYPE)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=DTYPE)
        if self.group_index >= len(data)-1:
            self.epoch += 1
            self.data_index = 0
            self.group_index = 0
        
        if self.data_index == len(data[self.group_index]):
            self.data_index = 0
            self.group_index = self.group_index + 1
        
        # locate group
        data_ = data[self.group_index]
        i = 0
        while True:
            input_ = data_[self.data_index]
            if len(data_)-1 > self.batch_size-i: context_words = random.sample([data_[w] for w in range(len(data_)) if w != self.data_index], self.batch_size-i)
            else: context_words = [w for w in data_ if w != input_]
            for context_word in context_words:
                batch[i] = input_
                labels[i, 0] = context_word
                i += 1
                if i >= self.batch_size:
                    self.data_index += 1
                    return batch, labels
            self.data_index += 1
            if self.data_index == len(data_):
                if self.group_index < len(data)-1: self.group_index += 1
                else:
                    self.group_index = 0
                    self.epoch += 1
                data_ = data[self.group_index]
                self.data_index = 0