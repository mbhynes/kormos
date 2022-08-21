kormos
=================================

The `kormos` package provides an interface between `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_ and `keras <https://keras.io>`_ for training models with deterministic minimization algorithms like `L-BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_.

It provides [peculiar] `keras` users with:

- `keras.Model` subclasses that may be optimized without changes in the API---a model may be trained using *either* the built-in stochastic mini-batch algorithms *or* the deterministic batch algorithms from `scipy.optimize`
- Out-of-the-box interoperability with the usual `keras` utilities and workflows; e.g.:

  - ``fit()`` still returns a the history object with optimization metadata and validation metrics from each iteration of the solver and is usable by `KerasTuner <https://keras.io/keras_tuner/>`_
  - Support for distributed training strategies (at least in principle---this has admittedly not been integration tested)
- The ability to use *second order* optimization methods from `scipy` by evaluating Hessian-vector-products if you have a very specific need for this (spoiler: you almost certainly do not)

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Motivation
-----------
Why would anyone want to go full batch in this day and age?

`Keras` is a powerful tool for developing predictive models and optimizing their parameters in a high-level language.
While its primary use case is large-scale deep learning, Keras's auto-differentiation utilities (from `Tensorflow`) that enable rapid prototyping and optimization with gradient-based minimization algorithms are great for other use cases in mathematical modelling and numerical optimization.

If you are working with models or datasets in which training data can reasonably fit in memory on a single machine, you may have situations in which deterministic algorithms like `L-BFGS <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_ or `Newton-CG <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html#optimize-minimize-newtoncg>`_ are complementary or viable alternatives to the *stochastic* `optimizers <https://keras.io/api/optimizers/>`_ available in `Keras`, since:

- deterministic algorithms don't require additional hyperparamter tuning to ensure convergence; if you're just prototyping something small and having trouble tuning learning rates, you may just want to crank out L-BFGS for a few minutes as a sanity check that your model can in fact be optimized
- these algorithms may have faster convergence to accurate solutions of the optimization problem if the dataset is small enough that full batch gradient and Hessian-vector-product computation times aren't prohibitive

So TL;DR: because Luddites exist even in the field of numerical optimization.

.. Luddites or Wackos lol? https://www.youtube.com/watch?v=0C4yBk6syOE#t=1m48s

Why The Name Kormos?
--------------------

Because `keras` is a powerful and useful tool, and is named after the Greek word *κέρας*, which means *horn*.

This package is related to `keras`, but it's not very powerful or useful.
It's named after the Greek word *κορμός*, which means *stump*.

License 
=======
This project is released under the MIT license, and contains adaptations of other codes released under the Apache and MIT licenses.
Please see the header in each source file for the applicable license and copyright notices therein. 

Setup
=====

Requirements
------------

The `kormos` package is built for:

- Python 3+ 
- Tensorflow 2+ (and the respective `keras` engine packaged with it)
- Scipy 0.1+ (any version really, since the `scipy.optimize.minimize` signature is stable)

Installation
------------

Install via the PyPI package [kormos](https://pypi.org/project/kormos/) using `pip`:

.. code-block:: python

  pip3 install kormos

Alternatively, if you like your releases bloody rare you may install from `git` directly:

.. code-block:: python

  pip3 install git+https://github.com/mbhynes/kormos

Usage Examples
==============

A `kormos` model is drag-and-drop replaceable with any `keras` model.
Below we provide some toy code examples, including Collaborative Filtering and MNIST classification examples adapted from the *Code Examples* section of `keras.io <https://keras.io/examples/>`_. 

Example: Linear Regression with Sequential API
----------------------------------------------

.. code-block:: python

  import numpy as np
  from tensorflow import keras

  import kormos

  rank = 50

  # Define the model using the keras.model.Model Sequential API
  model = kormos.models.BatchOptimizedSequentialModel()
  model.add(
      keras.layers.Dense(
          units=1,
          input_shape=(rank,),
          activation=None,
          use_bias=False,
          kernel_regularizer=keras.regularizers.L2(1e-3),
          kernel_initializer="ones",
      )
  )
  loss = keras.losses.MeanSquaredError()
  model.compile(loss=loss, optimizer='l-bfgs-b', metrics=['mean_absolute_error'])

  # Generate samples of normally distributed random data
  np.random.seed(1)
  w = np.random.normal(size=rank)
  X = np.random.normal(size=(1000, rank))
  y = np.expand_dims(X.dot(w), axis=1)

  Xval = np.random.normal(size=(1000, rank))
  yval = np.expand_dims(Xval.dot(w), axis=1)

  # Fit the model
  history = model.fit(
      x=X,
      y=y,
      epochs=10,
      validation_data=(Xval, yval),
      options={"maxcors": 3}, # can pass options payload if so desired
  )
  best_fit_weights = np.reshape(model.trainable_weights[0].numpy(), (1, -1))
  assert np.allclose(best_fit_weights, w, 1e-2)

We can now inspect the optimization metris traced in the ``history`` object returned from ``fit()``.
The training metrics captured by `kormos` include the:

- training loss function value (including regularization terms)
- 2-norm of the batch gradient
- number of evaluations of the loss/gradient function (equivalent to an *epoch* for a stochastic optimizer)
- number of evaluations of the Hessian-vector-product function, if applicable (equivalent to an *epoch* for a stochastic optimizer)

.. code-block:: python

  >>> import pandas as pd; pd.DataFrame(history.history)
          loss       grad  fg_evals  hessp_evals   val_loss  val_mean_absolute_error
  0  79.121972  17.946233         2            0  78.418121                 7.137860
  1   0.192005   0.713242         3            0   0.232164                 0.344657
  2   0.056429   0.186013         4            0   0.059140                 0.088700
  3   0.047397   0.042760         5            0   0.047348                 0.015531
  4   0.047006   0.008019         6            0   0.047006                 0.006401
  5   0.046991   0.001854         7            0   0.046994                 0.005846
  6   0.046990   0.000350         8            0   0.046992                 0.005675
  7   0.046990   0.000073         9            0   0.046992                 0.005642
  8   0.046990   0.000051        11            0   0.046992                 0.005642

We can now also *recompile* the model to use a stochastic optimizer; let's refit the model using ADAM:

.. code-block:: python

  # Recompile the model to use a different optimizer (this doesn't change its weights)
  model.compile(loss=model.loss, optimizer='adam', metrics=['mean_absolute_error'])

  # Reset the weights
  model.set_weights([np.random.random(size=(rank, 1))])

  # Fit the model using ADAM
  history = model.fit(
      x=X,
      y=y,
      epochs=150,
      validation_data=(Xval, yval),
  )

This is a somewhat contrived example in modern machine learning (small dataset and simple model with very few parameters), but it's the kind of classical use case in which a deterministic algorithm will converge faster than a stochastic algorithm. If you were interested in `keras` primarily for the nice `tensorflow` API and autodifferentiation routines, but had unsexy, non-deep modelling goals, this bud's for you:

.. code-block:: python

  >>> import pandas as pd; pd.DataFrame(history.history)
            loss  mean_absolute_error   val_loss  val_mean_absolute_error
  0    59.751369             6.218111  52.518566                 5.756832
  1    50.042812             5.688218  45.344589                 5.346300
  2    43.674156             5.308869  40.368832                 5.043641
  3    39.074280             5.021304  36.492527                 4.795147
  4    35.389912             4.781666  33.423710                 4.588754
  ..         ...                  ...        ...                      ...
  145   0.047031             0.008966   0.047031                 0.009047
  146   0.047023             0.008606   0.047025                 0.008718
  147   0.047017             0.008268   0.047019                 0.008344
  148   0.047012             0.007934   0.047013                 0.007977
  149   0.047008             0.007655   0.047009                 0.007717

  [150 rows x 4 columns]
    

Example: Linear Regression using the Functional API
---------------------------------------------------

The same linear regression model as above may be expressed equivalently by the functional API.
Here we specify a different `scipy` solver, the Newton-CG algorithm that uses Hessian-vector-products:

.. code-block:: python

  # Define the model using the keras.model.Model functional API
  model_input = keras.Input(shape=(rank,), name="input")
  model_output = keras.layers.Dense(
      units=1,
      input_shape=(rank,),
      activation=None,
      use_bias=False,
      kernel_regularizer=keras.regularizers.L2(1e-3),
      kernel_initializer="ones",
  )(model_input)
  model = kormos.models.BatchOptimizedModel(
      inputs=model_input,
      outputs=model_output,
  )
  loss = keras.losses.MeanSquaredError()
  model.compile(loss=loss, optimizer='newton-cg', metrics=['mean_absolute_error'])

  # Fit the model on the same data as previously
  history = model.fit(
      x=X,
      y=y,
      epochs=10,
      validation_data=(Xval, yval),
  )
  best_fit_weights = np.reshape(model.trainable_weights[0].numpy(), (1, -1))
  assert np.allclose(best_fit_weights, w, 1e-2)

The Newton-CG algorithm has second order convergence, so we should find that the gradient norm has decreased by several orders of magnitude more than with the L-BFGS-B algorithm.
(Of course, practically speaking this is a moot point in the world of approximate parameter estimation due to the limitations of both imperfect models and sampling bias that exists in training datasets: the numerical error in the solution is orders of magnitude smaller than other errors...)

Example: Collaborative Filtering for Item Recommendation
--------------------------------------------------------

We present a simple linear matrix factorization model for building a recommender system using the MovieLens dataset, and use the same preprocessing steps as in the `keras` example, `Collaborative Filtering for Movie Recommendations <https://keras.io/examples/structured_data/collaborative_filtering_movielens/>`_.

**Define the Model**

We define a simple matrix factorization model for factorizing the ratings matrix into the product of 2 latent feature matrices, represented by *user* and *item* embeddings: 

.. code-block:: python

  import tensorflow as tf
  from tensorflow import keras
  import kormos

  def build_model(rank, num_users, num_items, **kwargs):
      inputs = [
          keras.Input(shape=(1,), name="user", dtype=tf.int32),
          keras.Input(shape=(1,), name="item", dtype=tf.int32),
      ] 
      user_embedding = keras.layers.Embedding(
          input_dim=(num_users + 1),
          output_dim=rank,
          mask_zero=True,
          embeddings_initializer="normal",
          embeddings_regularizer=keras.regularizers.L2(1e-5),
          name="user_embedding",
      )
      item_embedding = keras.layers.Embedding(
          input_dim=(num_items + 1),
          output_dim=rank,
          mask_zero=True,
          embeddings_initializer="normal",
          embeddings_regularizer=keras.regularizers.L2(1e-5),
          name="item_embedding",
      )
      features = [
          user_embedding(inputs[0]),
          item_embedding(inputs[1]),
      ]
      output = keras.layers.Dot(axes=2, normalize=False)(features)
      model = kormos.models.BatchOptimizedModel(
          inputs=inputs,
          outputs=output,
          **kwargs
      )
      return model

**Prepare the Data**

We run the same pre-processing steps as in the `keras` example above.
(Please be aware that there are methodological errors in these steps that we have left unchanged: (1) it is not correct to split the training and testing data uniformly randomly, since some movies have only 1 rating and hence should not be members of the testing set, and (2) it is not possible to construct a factorization model that represents each user/item by a vector of rank ``k`` if ``k`` is *greater* than the number of observations (ratings) that that user/item has in the training data---such a system is `overdetermined <https://en.wikipedia.org/wiki/Overdetermined_system>`_).

.. code-block:: python

  import pandas as pd
  import numpy as np
  from zipfile import ZipFile
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers
  from pathlib import Path

  # Download the data from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
  # Use the ratings.csv file
  movielens_data_file_url = (
      "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
  )
  movielens_zipped_file = keras.utils.get_file(
      "ml-latest-small.zip", movielens_data_file_url, extract=False
  )
  keras_datasets_path = Path(movielens_zipped_file).parents[0]
  movielens_dir = keras_datasets_path / "ml-latest-small"

  # Only extract the data the first time the script is run.
  if not movielens_dir.exists():
      with ZipFile(movielens_zipped_file, "r") as zip:
          # Extract files
          print("Extracting all the files now...")
          zip.extractall(path=keras_datasets_path)
          print("Done!")

  ratings_file = movielens_dir / "ratings.csv"
  df = pd.read_csv(ratings_file)

  user_ids = df["userId"].unique().tolist()
  user2user_encoded = {x: i for i, x in enumerate(user_ids)}
  userencoded2user = {i: x for i, x in enumerate(user_ids)}
  movie_ids = df["movieId"].unique().tolist()
  movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
  movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
  df["user"] = df["userId"].map(user2user_encoded)
  df["movie"] = df["movieId"].map(movie2movie_encoded)

  num_users = len(user2user_encoded)
  num_movies = len(movie_encoded2movie)
  df["rating"] = df["rating"].values.astype(np.float32)
  # min and max ratings will be used to normalize the ratings later
  min_rating = min(df["rating"])
  max_rating = max(df["rating"])

  print(
      "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
          num_users, num_movies, min_rating, max_rating
      )
  )

  df = df.sample(frac=1, random_state=42)
  x = df[["user", "movie"]].values
  # Normalize the targets between 0 and 1. Makes it easy to train.
  y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
  # Assuming training on 90% of the data and validating on 10%.
  train_indices = int(0.9 * df.shape[0])
  x_train, x_val, y_train, y_val = (
      x[:train_indices],
      x[train_indices:],
      y[:train_indices],
      y[train_indices:],
  )

**Train the Model**

We may now train our factorization model:

.. code-block:: python

  rank = 5
  model = build_model(rank, num_users, num_movies)
  model.compile(
      loss=tf.keras.losses.MeanSquaredError(),
      optimizer="l-bfgs-b",
  )

  history = model.fit(
    x=(x_train[:, 0], x_train[:, 1]),
    y=y_train,
    batch_size=2**14,
    epochs=10,
    verbose=1
    validation_data=((x_val[:, 0], x_val[:, 1]), y_val),
  )

.. code-block:: python

  >>> import pandas as pd; pd.DataFrame(history.history)
          loss      grad  fg_evals  hessp_evals  val_loss
  0   0.499431  0.001055         2            0  0.497424
  1   0.492091  0.010318         5            0  0.496749
  2   0.491067  0.015367         7            0  0.499127
  3   0.461140  0.012731         9            0  0.472772
  4   0.271020  0.017515        12            0  0.327173
  5   0.228658  0.021585        14            0  0.298120
  6   0.156481  0.012698        16            0  0.226349
  7   0.125350  0.007833        17            0  0.193145
  8   0.101411  0.007957        18            0  0.169513
  9   0.093375  0.013233        19            0  0.162208
  10  0.082876  0.005307        20            0  0.152423
  11  0.077789  0.004717        21            0  0.149731
  12  0.072867  0.004420        22            0  0.144979
  13  0.066927  0.006463        23            0  0.137852
  14  0.063850  0.004983        24            0  0.136306
  15  0.061897  0.002353        25            0  0.133633
  16  0.060514  0.001867        26            0  0.132471
  17  0.058629  0.002211        27            0  0.131402
  18  0.057408  0.003710        28            0  0.130704
  19  0.056111  0.001484        29            0  0.129850
 

Example: MNIST convnet
----------------------

As a more realistic example of using `kormos` on a canonical dataset, we adapt the sample classification problem from the `MNIST convnet <https://keras.io/examples/vision/mnist_convnet/>`_ example.
Please note that this convolutional network model has a large number of highly correlated parameters to optimize, and stochastic algorithms like ADAM will generally perform better and provide better results.
However we provide it as an example of how both stochastic and deterministic algorithms may be combined by *recompiling* a `kormos` model.

**Prepare the Data**

.. code-block:: python

  import numpy as np

  from tensorflow import keras 
  from keras import layers

  # Model / data parameters
  num_classes = 10
  input_shape = (28, 28, 1)

  # Load the data and split it between train and test sets
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

  # Scale images to the [0, 1] range
  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255
  # Make sure images have shape (28, 28, 1)
  x_train = np.expand_dims(x_train, -1)
  x_test = np.expand_dims(x_test, -1)
  print("x_train shape:", x_train.shape)
  print(x_train.shape[0], "train samples")
  print(x_test.shape[0], "test samples")

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

**Build the Model**

.. code-block:: python
  
  from kormos.models import BatchOptimizedSequentialModel

  def build_model():
      model = BatchOptimizedSequentialModel(
          [
              keras.Input(shape=input_shape),
              layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
              layers.MaxPooling2D(pool_size=(2, 2)),
              layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
              layers.MaxPooling2D(pool_size=(2, 2)),
              layers.Flatten(),
              layers.Dropout(0.5),
              layers.Dense(num_classes, activation="softmax"),
          ]
      )
      return model

  model = build_model()
  model.summary()

.. code-block::

  Model: "batch_optimized_sequential_model"
  _________________________________________________________________
   Layer (type)                   Output Shape              Param #
  =================================================================
   conv2d (Conv2D)                (None, 26, 26, 32)        320

   max_pooling2d (MaxPooling2D)   (None, 13, 13, 32)        0

   conv2d_1 (Conv2D)              (None, 11, 11, 64)        18496

   max_pooling2d_1 (MaxPooling2D) (None, 5, 5, 64)          0

   flatten (Flatten)              (None, 1600)              0

   dropout (Dropout)              (None, 1600)              0

   dense (Dense)                  (None, 10)                16010

  =================================================================
  Total params: 34,826
  Trainable params: 34,826
  Non-trainable params: 0
  _________________________________________________________________

**Train the Model**

We use this example train the model by running a combination of different algorithms.
We start by running ADAM for 1 epoch, and then using this solution as a warm start initial guess for a batch solver by *recompiling* the model:

.. code-block:: python

  loss = keras.losses.CategoricalCrossentropy()
  # Train a model with ADAM
  model = build_model()
  model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
  hist1 = model.fit(x_train, y_train, batch_size=2**5, epochs=1, validation_data=(x_test, y_test))

  # Continue training the model with a batch algorithm.
  # We can instantiate the optimizer as well instead of a string identifier
  optimizer = kormos.optimizers.ScipyBatchOptimizer()
  model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

  # We can specify the method and any options for it in fit as keyword wargs
  hist2 = model.fit(
      x_train,
      y_train,
      batch_size=2**14, # this is much larger than for stochastic solvers!
      epochs=3,
      validation_data=(x_test, y_test),
      method='bfgs',
  )


Implementation Details
======================

The `kormos` package implements an interface for batch optimization and wraps `scipy.optimize.minimize` in that interface in the following steps:

- We create a subclass of `keras.Model`, `BatchOptimizedModel` (and `BatchOptimizedSequentialModel` to extend the `Sequential` API).

- The subclass provides a ``fit_batch()`` method with nearly identical signature to the parent ``fit()``, but does not perform stochastic mini-batch optimization. Instead, this method offloads all optimization to the the model's ``optimizer`` attribute, which must implement the method ``minimize()`` to perform training by minimizing the the loss function provided during model compilation.

- When a `BatchOptimizedModel` is compiled with a `BatchOptimzer` (or string identifier for one) as its `optimizer` argument, the ``fit`` method inherited from `keras.Model` is overriden with a pointer to ``fit_batch()`` (such that a `BatchOptimizedModel` may be trained with either stochastic or deterministic solvers, depending on how it's compiled).

- The `ScipyBatchOptimizer` class extends the `BatchOptimizer` interface and uses the `scipy.optimize.minimize` routine to fit the model.

At first face this is more complicated than the *recommended* way of extending `keras` to perform custom training (i.e. by overriding ``keras.Model.train_step`` such as in the article `Customizing what happens in fit() <https://keras.io/guides/customizing_what_happens_in_fit/>`_).
However, unfortunately we found extending ``train_step`` to be awkward or infeasible for implementing a batch optimization algorithm while still making use of the standard `keras` utilities for computing *validation metrics* at each iteration end (epoch).
Overriding the model ``train_step`` (and putting the call to `scipy.optimize.minimize` inside it) would mean that from the `keras` model's perspective only a single *epoch* would be performed, such that validation metrics would only be computed at the very end of the optimzation routine.

Acknowledgements & Related Work
================================

This package has adapted code from the following sources:

- `Pi-Yueh Chuang's <https://pychao.com/contact-us-and-pgp-key/>`_ MIT-licensed `scipy.optimize.minimize_lbfgs` wrapper available on github `here <https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993>`_.
- `Allen Lavoie's <https://github.com/allenlavoie>`_ Hessian-vector-product routines from `tensorflow`, available on github `here <https://github.com/tensorflow/tensorflow/commit/5b37e7ed14eb7dddae8a0e87435595347a315bb7>`_ under the Apache License version 2.

There is also a related project `keras-opt <https://github.com/pedro-r-marques/keras-opt>`_ with the same goal but different implementation and API.
The `kormos` package is recommended over `keras-opt` because its implementation is faster and more robust when training models with large memory requirements, it exposes all of the arguments to `scipy.optimize.minimize` if you wish to solve a constrained optimization problem, and is a little bit more seemless to use as part of the native `keras` workflow.

