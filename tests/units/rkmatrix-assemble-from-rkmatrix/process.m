clear all;
load rkmatrix-assemble-from-rkmatrix.output;

norm(M_rk.A * M_rk.B' - M, 'fro') / norm(M, 'fro')
