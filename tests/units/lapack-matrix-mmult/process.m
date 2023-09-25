clear all;
load lapack-matrix-mmult.output;

C_octave = A * B;
norm(C_octave - C, 'fro') / norm(C_octave, 'fro')
norm(C_adding_before + C_octave - C_adding, 'fro') / norm(C_adding, 'fro')
