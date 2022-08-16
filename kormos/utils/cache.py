# MIT License
#
# Copyright (c) 2022 Michael B Hynes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import functools

import numpy as np
import tensorflow as tf


class OptimizationStateCache(object):
    def __init__(self, max_entries=100):
        assert max_entries > 0
        self.max_entries = max_entries
        self.state_dict = {}
        self.entries = []

    @staticmethod
    def hash(x):
        if type(x) is tf.Tensor:
            h = OptimizationStateCache.hash(x.numpy())
        if type(x) is np.ndarray:
            h = hex(hash(x.tobytes()))
        else:
            h = x
        return h

    def clear(self):
        del self.state_dict
        del self.entries
        self.state_dict = {}
        self.entries = []

    def _delete_oldest_entry(self):
        if len(self.entries) == self.max_entries:
            self.entries.reverse()
            state_key = self.entries.pop()
            self.entries.reverse()
            del self.state_dict[state_key]

    def update(self, x, **kwargs):
        state_key = self.hash(x)

        if state_key not in self.state_dict:
            self._delete_oldest_entry()
            self.entries.append(state_key)

        state = self.state_dict.pop(state_key, {"x": copy.deepcopy(x)})
        state.update({key: copy.deepcopy(val) for (key, val) in kwargs.items()})
        self.state_dict[state_key] = state

    def get(self, x, key):
        state_key = self.hash(x)
        state = self.state_dict.get(state_key, {})
        return state.get(key)

    def get_last(self, key, k=-1):
        state_key = self.entries[k]
        state = self.state_dict.get(state_key, {})
        return state.get(key)

    @staticmethod
    def cached(key, max_entries=10, cache_name="cache"):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, x, *args, **kwargs):
                cache = getattr(self, cache_name)
                cached_value = cache.get(x, key)
                if cached_value is not None:
                    return cached_value
                value = func(self, x, *args, **kwargs)
                cache.update(x, **{key: value})
                return value

            return wrapper

        return decorator
