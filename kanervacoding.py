#!/usr/bin/env python
from __future__ import print_function
import numpy as np

class kanervacoder:
  def __init__(self, dims, ptypes, sparsity, limits, dist=lambda x1, x2: np.max(np.abs(x1 - x2), axis=1), seed=None):
    np.random.seed(seed)
    self._n_dims = dims
    self._n_pts = ptypes
    self._k = int(round(sparsity * ptypes))
    self._lims = np.array(limits)
    self._ranges = self._lims[:, 1] - self._lims[:, 0]
    self._pts = np.random.random([self._n_pts, self._n_dims])
    self._dist = dist

  @property
  def n_ptypes(self):
    return self._n_pts
  
  def __getitem__(self, x):
    xs = (x - self._lims[:, 0]) / self._ranges
    return np.argpartition(self._dist(self._pts, xs), self._k)[:self._k]


def example():
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  import time

  # kanerva coder dimensions, limits, prototypes, sparsity
  dims = 2
  lims = [(0, 2.0 * np.pi)] * 2
  ptypes = 512
  sparsity = 0.025

  # create kanerva coder
  K = kanervacoder(dims, ptypes, sparsity, lims)

  # learning params
  w = np.zeros(K.n_ptypes)
  alpha = 0.1 / round(sparsity * ptypes)

  # target function with gaussian noise
  def target_ftn(x, y, noise=True):
    return np.sin(x) + np.cos(y) + noise * np.random.randn() * 0.1

  # randomly sample target function until convergence
  timer = time.time()
  batch_size = 100
  for iters in range(100):
    mse = 0.0
    for b in range(batch_size):
      xi = lims[0][0] + np.random.random() * (lims[0][1] - lims[0][0])
      yi = lims[1][0] + np.random.random() * (lims[1][1] - lims[1][0])
      zi = target_ftn(xi, yi)
      phi = K[xi, yi]
      w[phi] += alpha * (zi - w[phi].sum())
      mse += (w[phi].sum() - zi) ** 2
    mse /= batch_size
    print('samples:', (iters + 1) * batch_size, 'batch_mse:', mse)
  print('elapsed time:', time.time() - timer)

  # get learned function
  print('mapping function...')
  res = 200
  x = np.arange(lims[0][0], lims[0][1], (lims[0][1] - lims[0][0]) / res)
  y = np.arange(lims[1][0], lims[1][1], (lims[1][1] - lims[1][0]) / res)
  z = np.zeros([len(y), len(x)])
  for i in range(len(x)):
    for j in range(len(y)):
      z[i, j] = w[K[x[i], y[j]]].sum()

  # plot
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X, Y = np.meshgrid(x, y)
  surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
  plt.show()

if __name__ == '__main__':
  example()
