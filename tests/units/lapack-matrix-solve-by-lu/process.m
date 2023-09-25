clear all;
load lapack-matrix-solve-by-lu.output;

x_octave = M \ b;
norm(x_octave - x, 2) / norm(x_octave, 2)
