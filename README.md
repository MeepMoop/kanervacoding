# Kanerva Coding

[Kanerva coding](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node88.html#SECTION04234000000000000000) is a function approximation method that distributes *prototypes*  across the state space, and learns weights associated with them. This implementation is a variant that uses the L∞ distance metric, and activates the *k*-nearest prototypes (set through a sparsity parameter) to the input.

# Dependencies

* numpy
* matplotlib (to run the example)

# Usage

```python
import numpy as np
from kanervacoding import kanervacoder

# dimensions, value limits of each dimension, prototypes, and sparsity
dims = 4
lims = [(3.0, 7.5), (-4.4, 4.2), (9.6, 12.7), (0.0, 1.0)]
ptypes = 1024
sparsity = 0.025  # (k = sparsity * ptypes)

# create kanervacoder
K = kanervacoder(dims, ptypes, sparsity, lims)

# init weights and step size
theta = np.zeros(K.n_ptypes)
alpha = 0.1 / round(sparsity * ptypes)

# training iteration with value 5.5 at location (3.3, -2.1, 11.1, 0.7)
phi = K[3.3, -2.1, 11.1, 0.7]
theta[phi] = alpha * (5.5 - theta[phi].sum())

# get approximated value at (3.3, -2.1, 11.1, 0.7)
print(theta[phi].sum())
```

# Examples
<p align="center">
  <img src="https://github.com/MeepMoop/kanervacoding/blob/master/examples/kc_sincos.png"><br>
  Kanerva coder with 512 prototypes and 2.5% sparsity approximating f(x, y) = sin(x) + cos(y) + <i>N</i>(0, 0.1)<br><br>
</p>
