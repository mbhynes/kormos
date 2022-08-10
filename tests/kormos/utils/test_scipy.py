import numpy as np
import scipy.optimize as opt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import L1, L2
from tensorflow.python.keras.engine import data_adapter

from kormos.utils.scipy import FunctionFactory


class TestFunctionFactory(object):

  keras.backend.set_floatx("float64")

  def _build_ols_model(self, rank=3):
    w = 1 + np.arange(rank)
    X = np.ones(shape=(2, rank))
    y = X.dot(w)
    sample_weight = np.array([4.0, 3.0]) # different weights, but should be normalized out
    dataset = tf.data.Dataset.from_tensor_slices(
      tensors=(tf.convert_to_tensor(X), tf.convert_to_tensor(y), tf.convert_to_tensor(sample_weight))
    )

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
      units=1,
      input_shape=(rank,),
      activation=None,
      use_bias=False,
      kernel_regularizer=L2(1e-1),
      kernel_initializer="ones",
    ))
    model.compile(loss=keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM), optimizer="adam")
    return model, dataset

  def _build_mlp_model(self, train_size=(10, 3)):
    w = np.random.normal(size=train_size[-1])
    X = tf.convert_to_tensor(np.random.normal(size=train_size), dtype=tf.dtypes.float64)
    y = tf.convert_to_tensor((X.numpy().dot(w) > 0.25).astype(int), dtype=tf.dtypes.float64)
    dataset = tf.data.Dataset.from_tensor_slices(tensors=(X, y))

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
      units=3,
      input_shape=(train_size[-1],),
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
    model.compile(loss=keras.losses.BinaryCrossentropy(reduction=keras.losses.Reduction.SUM), optimizer="adam")
    return model, dataset

  def test_grad_analytic(self):
    rank = 3
    model, dataset = self._build_ols_model(rank)
    factory = FunctionFactory(
      model,
      Xyw=dataset,
      dtype=tf.dtypes.float64,
      batch_size=1,
    ).build()

    # Get the first row from the dataset with a non-zero sample weight
    rows = list(dataset.as_numpy_iterator())
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    print(X, y)

    # Retrieve the model weights, which should be the same as the all-ones initial values
    w = factory.get_weights()
    assert w.sum() == len(w)
    bias = 0.0
    y_pred = X.dot(w) + bias 
    reg_coeff = 0.1
    n = X.shape[0]
    expected_loss = (1 / n) * (y - y_pred).dot(y - y_pred) + reg_coeff * w.dot(w)
    expected_grad = (2 / n) * X.T.dot(y_pred - y) + 2 * (reg_coeff * w)
    expected_hess = (2 / n) * X.T.dot(X) + 2 * reg_coeff * np.eye(rank)

    assert factory.func(w) == expected_loss, f"factory.func(w) = {factory.func(w)} != expected = {expected_loss}"
    assert np.allclose(factory.grad(w), expected_grad), f"factory.grad = {factory.grad(w)} did not match expected = {expected_grad}"
    for k in range(rank):
      p = np.eye(rank)[k]
      assert np.allclose(
        factory.hessp(w, p),
        expected_hess[:, k],
      ), f"factory.hessp for {p} = {factory.hessp(w, p)} did not match expected = {expected_hess[:, k]}"

  def test_grad_numeric(self):
    tol = 1e-4
    model, dataset = self._build_mlp_model()
    factory = FunctionFactory(
      model,
      Xyw=dataset,
      dtype=tf.dtypes.float64,
      batch_size=2,
    ).build()
    x = np.array(0 * factory.get_weights() + 1.0, dtype=np.float64)
    err = opt.check_grad(factory.func, factory.grad, x)
    if err > tol:
      g_approx = opt.approx_fprime(x, factory.func, epsilon=1e-7)
      g = factory.grad(x)
      assert err <= tol, f"Analytical gradient and numerical gradient differ by {err} > {tol}:\n grad: {g}\ngrad_approx: {g_approx}"

  def test_hessp_numeric(self):
    tol = 1e-4
    model, dataset = self._build_mlp_model()
    factory = FunctionFactory(
      model,
      Xyw=dataset,
      dtype=tf.dtypes.float64,
      batch_size=2,
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
