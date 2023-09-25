clear all;
load rkmatrix-agglomeration-of-two-submatrices-rank0.output;

norm(M_agglomerated1_rk.A * M_agglomerated1_rk.B' - M_agglomerated1, 'fro') / norm(M_agglomerated1, 'fro')
norm(M_agglomerated2_rk.A * M_agglomerated2_rk.B' - M_agglomerated2, 'fro') / norm(M_agglomerated2, 'fro')
