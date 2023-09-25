clear all;
load lapack-matrix-agglomeration.output;

M_from_octave = [M11, M12; M21, M22]
norm(M - M_from_octave, "fro") / norm(M, "fro")
