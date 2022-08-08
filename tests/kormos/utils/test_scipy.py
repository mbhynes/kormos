import numpy as np
import scipy.optimize as opt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import L1, L2

from kormos.utils.scipy import FunctionFactory


class TestFunctionFactory(object):

  keras.backend.set_floatx("float64")

  def _build_model(self, train_size=(10, 3)):
    w = np.random.normal(size=train_size[1])
    X = tf.convert_to_tensor(np.random.normal(size=train_size), dtype=tf.dtypes.float64)
    y = tf.convert_to_tensor((X.numpy().dot(w) > 0.25).astype(float), dtype=tf.dtypes.float64)
    dataset = tf.data.Dataset.from_tensor_slices(tensors=(X, y))

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
      units=3,
      input_shape=(train_size[1],),
      activation='sigmoid',
      use_bias=False,
      kernel_regularizer=L1(1e-1),
      kernel_initializer="ones",
    ))
    model.add(keras.layers.Dense(
      units=1,
      activation='sigmoid',
      use_bias=False,
      kernel_regularizer=L2(1e-1),
      kernel_initializer="ones",
    ))
    model.compile(loss="binary_crossentropy", optimizer="adam")

    return model, dataset

  def test_grad(self):
    tol = 1e-4
    model, dataset = self._build_model()
    factory = FunctionFactory(
      model,
      keras.losses.BinaryCrossentropy(from_logits=False),
      Xyw=dataset,
      dtype=tf.dtypes.float64,
      batch_size=3,
    ).build()
    x = np.array(0 * factory.get_weights() + 1.0, dtype=np.float64)
    err = opt.check_grad(factory.func, factory.grad, x)
    if err > tol:
      g_approx = opt.approx_fprime(x, factory.func, epsilon=1e-7)
      g = factory.grad(x)
      assert err <= tol, f"Analytical gradient and numerical gradient differ by {err} > {tol}:\n grad: {g}\ngrad_approx: {g_approx}"

  def test_hessp(self):
    tol = 1e-4
    model, dataset = self._build_model()
    factory = FunctionFactory(
      model,
      keras.losses.BinaryCrossentropy(from_logits=False),
      Xyw=dataset,
      dtype=tf.dtypes.float64,
      batch_size=3,
    ).build()
    x = np.array(0 * factory.get_weights() + 1.0, dtype=np.float64)

    def _hess(x):
      n = len(x)
      H = np.zeros((n, n))
      grad = lambda x: opt.approx_fprime(x, factory.func, epsilon=1e-5)
      for k in range(n):
        h_approx = opt.approx_fprime(x, lambda x: grad(x)[k], epsilon=1e-5)
        H[k, :] = h_approx
      return H
      
    H = _hess(x)
    p = np.zeros(len(x))
    for k in range(len(p)):
      p[k] = 1
      Hp = H.dot(p)
      hessp = factory.hessp(x, p)
      assert np.allclose(Hp, hessp, atol=1e-2), f"hessp does not match numerical evaluation for k={k}:\nhessp:{hessp}\napprox: {Hp}"
      p[k] = 0
