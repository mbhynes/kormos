import setuptools

with open("README.rst", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="kormos",
  version="0.1.4",
  author="Michael B Hynes",
  author_email="mike.hynes.rhymes@gmail.com",
  description="An interface to scipy.optimize.minimize for training Keras models with batch optimization algorithms like L-BFGS.",
  long_description=long_description,
  url="https://github.com/mbhynes/kormos",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Information Analysis",
  ],
  python_requires='>=3',
  install_requires=[
    "tensorflow>=2",
    "numpy",
    "scipy",
  ],
)
