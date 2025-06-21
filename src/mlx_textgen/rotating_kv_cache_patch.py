import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import RotatingKVCache, KVCache # Import necessary classes
from mlx.utils import tree_map


def new_update_and_fetch(self, k, v):
    KVCache.update_and_fetch(self, k, v)
    self._idx = self.keys.shape[2]
    return self.state


def new_state_getter(self):
    if self.offset <= self.max_size:
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
    elif self.keep:
        keys = mx.concat(self.keys[..., :self.keep, :], self.keys[..., (self.offset -(self.max_size - self.keep)):self.offset, :], axis=2)
        values = mx.concat(self.values[..., :self.keep, :], self.values[..., (self.offset -(self.max_size - self.keep)):self.offset, :], axis=2)
        return keys, values
    else:
        return self.keys[..., (self.offset - self.max_size):self.offset, :], self.values[..., (self.offset - self.max_size):self.offset, :]
    
new_state_setter = KVCache.state.fset

new_is_trimmable = KVCache.is_trimmable

new_trim = KVCache.trim
        



# --- Apply the Patches ---
RotatingKVCache.update_and_fetch = new_update_and_fetch
RotatingKVCache.state = property(new_state_getter, new_state_setter)
RotatingKVCache.trim = new_trim
RotatingKVCache.is_trimmable = new_is_trimmable


