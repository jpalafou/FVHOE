array_manager tests

PAD only?
20% local, relative, no blending

no limiting (advection of square, 2D, CPU vs GPU)
    p=0, euler vs p=0 (SD_update=True), m=0 # should be the same
    at different p [1, 3, 7], fixed number of steps
        compare m=0 SD to .euler() FV (exact same integrator) (nans, just for timing)
        compare m=1 SD to .ssprk2() FV
        compare m=2 SD to .ssprk3() FV
        compare m=3 SD to .rk4() FV
    SD_update=True # turns off fallback scheme
    ader = "old" vs. "new"
    fv.execution_time vs. sd.execution_time
repeat with limiting

convex blending in 3D:

1/4 1/4 1/4 1/4 1/4
1/4 1/2 3/4 1/2 1/4
1/4 3/4  1  3/4 1/4
1/4 1/2 3/4 1/4 1/4
1/4 1/4 1/4 1/4 1/4

1/4 1/4 1/4 1/4 1/4
1/4 3/8 1/2 3/8 1/4
1/4 1/2 3/4 1/2 1/4
1/4 3/8 1/2 3/8 1/4
1/4 1/4 1/4 1/4 1/4

1/4 1/4 1/4 1/4 1/4
1/4 1/4 1/4 1/4 1/4
1/4 1/4 1/4 1/4 1/4
1/4 1/4 1/4 1/4 1/4
1/4 1/4 1/4 1/4 1/4
