# sparse-interp
Toolbox for interpolating from an uneven grid.

Uses kd-tree implementation for fast nearest-neighbor search in Python.
Classes are written for a regularly sampled grid (e.g. Bayer array) or an
irregularly sampled grid. The input is a single channel with values at regular
or irregular grid points, and the output is multiple channel image with values
interpolated from whichever points correspond to that channel (given a lookup table).

Can be used as a differentiable PyTorch nn.Module.

## Getting started
Use the model in `model.py` to interpolate an image. An example is given there.