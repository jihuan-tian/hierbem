clear all;
load lapack-matrix-agglomeration-interwoven-indices.output;

M_from_octave = zeros(5, 5);
M_from_octave([3,5], [1,3,5]) = M11;
M_from_octave([3,5], [2,4]) = M12;
M_from_octave([1,2,4], [1,3,5]) = M21;
M_from_octave([1,2,4],[2,4]) = M22
norm(M - M_from_octave, "fro") / norm(M, "fro")
