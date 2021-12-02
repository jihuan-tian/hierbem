clear all;
load lapack-matrix-solve-by-cholesky.output;
load M.dat;

x_octave = M \ b;
norm(x_octave - x, 2) / norm(x_octave, 2)
