# infimnist_py
Python binding for the infinite MNIST dataset generator of L. Bottou (see http://leon.bottou.org/projects/infimnist), written in Cython. A tensorflow queue for real-time data generation is also provided in `infimnist_queue.py`.

# Installation

You need to copy (or symlink) the `data/` directory from the original infimnist project folder (available [here](http://leon.bottou.org/projects/infimnist)) into the root folder of this repo.

Build the cython extension with:
```
python setup.py build_ext -if
```

# Example usage
The following code gives the first test example (0), the first training example (10000) and its first transformation (70000). The indexing logic follows that of the original infimnist binary described [here](http://leon.bottou.org/projects/infimnist).
```python
import _infimnist as infimnist
import numpy as np

mnist = infimnist.InfimnistGenerator()
indexes = np.array([0, 10000, 70000], dtype=np.int64)
digits, labels = mnist.gen(indexes)

# example of preprocessing from [0, 255] to [0., 1.]
X = digits.astype(np.float32).reshape(indexes.shape[0], 28, 28)
X = X / 255

import matplotlib.pyplot as plt
plt.imshow(X[0])
plt.title('label: {}'.format(labels[0]))
```
