* add a submodule for stencilpal
* add coverage
* add docs
* conduct tests with PAD-only limiting
* implement convex blending in 3D:


~ ~ third layer ~ ~

1/4 1/4 1/4 1/4 1/4

1/4 1/2 3/4 1/2 1/4

1/4 3/4  1  3/4 1/4

1/4 1/2 3/4 1/4 1/4

1/4 1/4 1/4 1/4 1/4

~ ~ fourth layer~ ~

1/4 1/4 1/4 1/4 1/4

1/4 3/8 1/2 3/8 1/4

1/4 1/2 3/4 1/2 1/4

1/4 3/8 1/2 3/8 1/4

1/4 1/4 1/4 1/4 1/4

~ ~ fifth layer ~ ~

1/4 1/4 1/4 1/4 1/4

1/4 1/4 1/4 1/4 1/4

1/4 1/4 1/4 1/4 1/4

1/4 1/4 1/4 1/4 1/4

1/4 1/4 1/4 1/4 1/4

~ ~ ~ ~ ~ ~ ~ ~ ~ ~

* make arrays 5D: (var, x-coord, y-coord, z-coord, interpolation)
* MOOD loop
* move density and pressure floors to PAD
* slope limiting and sound speed should be computed from cell averages
