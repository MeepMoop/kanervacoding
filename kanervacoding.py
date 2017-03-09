#!/usr/bin/env python
from __future__ import print_function
import numpy as np

class kanervacoder:
  def __init__(self, dims, ptypes, sparsity, limits, step_size=0.1, seed=None):
    self._n_dims = dims
    self._n_pts = ptypes
    self._k = int(sparsity * ptypes)
    self._lims = np.array(limits)
    self._alpha = step_size / self._k
    self._pts = np.zeros([self._n_pts, self._n_dims]); np.random.seed(seed)
    for i in range(dims):
      self._pts[:, i] = self._lims[i][0] + (self._lims[i][1] - self._lims[i][0]) * np.random.random(self._n_pts)
    self._w = np.zeros(self._n_pts)
  
  def _get_active_pts(self, x):
    self._a_pts = np.argpartition(np.sum((self._pts - x) ** 2, axis=1), self._k)[:self._k]
  
  def __getitem__(self, x):
    self._get_active_pts(x)
    return np.sum(self._w[self._a_pts])

  def __setitem__(self, x, val):
    self._get_active_pts(x)
    self._w[self._a_pts] += self._alpha * (val - np.sum(self._w[self._a_pts]))

  def set_step_size(self, step_size):
    self._alpha = step_size / self._k

def example():
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  import time

  # kanerva coder dimensions, limits, tilings, step size, and offset vector
  dims = 2
  ptypes = 1024
  sparsity = 32 / 1024
  lims = [(0, 2.0 * np.pi)] * 2
  alpha = 0.1
  seed = None

  # create kanerva coder
  K = kanervacoder(dims, ptypes, sparsity, lims, alpha, seed)

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
      K[xi, yi] = zi
      mse += (K[xi, yi] - zi) ** 2
    mse /= batch_size
    print('samples:', (iters + 1) * batch_size, 'batch_mse:', mse)
  print('elapsed time:', time.time() - timer)

  # get learned function
  print('mapping function...')
  res = 100
  x = np.arange(lims[0][0], lims[0][1], (lims[0][1] - lims[0][0]) / res)
  y = np.arange(lims[1][0], lims[1][1], (lims[1][1] - lims[1][0]) / res)
  z = np.zeros([len(y), len(x)])
  for i in range(len(x)):
    for j in range(len(y)):
      z[j, i] = K[x[i], y[j]]

  # plot
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X, Y = np.meshgrid(x, y)
  surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
  plt.show()

if __name__ == '__main__':
  example()
