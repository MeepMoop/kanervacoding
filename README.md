# Kanerva Coding

[Kanerva coding](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node88.html#SECTION04234000000000000000) is a function approximation method that distributes *prototypes*  across the state space, and learns weights associated with them. This implementation uses a Euclidian distance measure, and activates the *k*-nearest prototypes to the input.

# Dependencies

* numpy
* matplotlib (to run the example)

# Usage

```python
from kanervacoding import kanervacoder

# dimensions, prototypes, and activation sparsity
dims = 4
ptypes = 1024
sparsity = 0.025  # (k = sparsity * prototypes)

# value limits of each dimension (min, max)
lims = [(3.0, 7.5), (-4.4, 4.2), (9.6, 12.7), (0.0, 1.0)]

# create kanervacoder with step size 0.1
K = kanervacoder(dims, ptypes, sparsity, lims, 0.1)

# training iteration with value 5.5 at location (3.3, -2.1, 11.1, 0.7)
K[3.3, -2.1, 11.1, 0.7] = 5.5

# get approximated value at (3.3, -2.1, 11.1, 0.7)
print K[3.3, -2.1, 11.1, 0.7]
```
