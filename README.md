# sparse-interp
Toolbox for interpolating from an uneven grid


Uses kd-tree implementation for fast nearest-neighbor search in Python.
Classes are written for a regularly sampled grid (e.g. Bayer array) or an
irregularly sampled image.

Can be used as a differentiable PyTorch nn.Module.