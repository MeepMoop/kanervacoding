# Kanerva Coding

[Kanerva coding](http://incompleteideas.net/book/ebook/node88.html#SECTION04234000000000000000) is a function approximation method that distributes *prototypes*  across the state space, and learns weights associated with them. This implementation is a variant that uses the L∞ distance metric, and activates the *k*-nearest prototypes to the input.

# Dependencies

* numpy
* matplotlib (to run the example)

# Usage

```python
import numpy as np
from kanervacoding import kanervacoder

# dimensions, value limits of each dimension, prototypes, and number of active features (k)
dims = 4
lims = [(3.0, 7.5), (-4.4, 4.2), (9.6, 12.7), (0.0, 1.0)]
ptypes = 1024
n_active = 32 

# create kanervacoder
K = kanervacoder(dims, ptypes, n_active, lims)

# init weights and step size
w = np.zeros(K.n_ptypes)
alpha = 0.1 / n_active

# training iteration with value 5.5 at location (3.3, -2.1, 11.1, 0.7)
phi = K[3.3, -2.1, 11.1, 0.7]
w[phi] = alpha * (5.5 - w[phi].sum())

# get approximated value at (3.3, -2.1, 11.1, 0.7)
print(w[phi].sum())
```

# Examples
<p align="center">
  <img src="https://github.com/MeepMoop/kanervacoding/blob/master/examples/kc_sincos.png"><br>
  Kanerva coder with 512 prototypes and 2.5% sparsity approximating f(x, y) = sin(x) + cos(y) + <i>N</i>(0, 0.1)<br><br>
</p>
