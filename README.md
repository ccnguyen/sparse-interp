# sparse-interp
Toolbox for interpolating from an uneven grid.

- Interpolation kd-tree implementation for fast nearest-neighbor search in Python.
Classes are written for a regularly sampled grid (e.g. Bayer array) or an
irregularly sampled grid. The input is a single channel with values at regular
or irregular grid points, and the output is multiple channel image with values
interpolated from whichever points correspond to that channel (given a lookup table).

Can be used as a differentiable PyTorch nn.Module.

- Multi-class Poisson disk sampling on a dense grid
Implemented as described in Wei 2009. 

## Getting started
Use the model in `model.py` to interpolate an image. An example is given there.

Use `multiclass_poisson.py` for the multi-class Poisson sampler.

### References
----------
```BibTex
@article{wei2010multi,
  title={Multi-class blue noise sampling},
  author={Wei, Li-Yi},
  journal={ACM Transactions on Graphics (TOG)},
  volume={29},
  number={4},
  pages={1--8},
  year={2010},
  publisher={ACM New York, NY, USA}
}

```
