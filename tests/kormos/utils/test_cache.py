import numpy as np

from kormos.utils.cache import OptimizationStateCache

  
class TestOptimizationStateCache(object):

  def test_hash(self):
    assert OptimizationStateCache.hash(np.zeros(0)) == "0x0"
    eps = np.finfo(np.float128).eps
    x1 = np.ones(10) + np.random.random(10)
    x2 = x1.copy()
    x3 = x1 + np.finfo(np.float64).eps * np.ones(x1.shape)
    x4 = x1 + (eps / 2)
    assert OptimizationStateCache.hash(x1) == OptimizationStateCache.hash(x2)
    assert OptimizationStateCache.hash(x1) != OptimizationStateCache.hash(x3)
    assert OptimizationStateCache.hash(x1) == OptimizationStateCache.hash(x4)

  def test_update(self):
    fn = lambda x: x.sum()
    n = 3
    x = np.eye(n)
    c = OptimizationStateCache(max_entries=(n - 1))
    for k in range(n):
      f = fn(x[k])
      c.update(x[k], f=fn(x[k]))
      assert c.get(x[k], 'f') == f
      assert c.get(x[k], 'invalid') == None

    assert c.get(x[0], 'f') == None
    assert set(c.state_dict.keys()) == {c.hash(x[k]) for k in range(1, n)}


class TestCachedDecorator(object):

  def __init__(self):
    self.cache = OptimizationStateCache()
    self.num_evals = 0

  @OptimizationStateCache.cached('cache_key')
  def _cached_method(self, x):
    self.num_evals += 1
    return 2 * x

  def test_decorated_method(self):
    assert self.cache.get(1, 'cache_key') == None
    assert self.num_evals == 0

    f = self._cached_method(1)
    assert f == 2
    assert self.cache.get(1, 'cache_key') == f
    assert self.num_evals == 1

    f2 = self._cached_method(1)
    assert f2 == f
    assert self.num_evals == 1
