clear all;
load rkmatrix-agglomeration-of-two-submatrices-interwoven-indices.output;

norm(M_agglomerated1 - M_agglomerated1_rk.A * M_agglomerated1_rk.B', 'fro') / norm(M_agglomerated1, 'fro')
norm(M_agglomerated2 - M_agglomerated2_rk.A * M_agglomerated2_rk.B', 'fro') / norm(M_agglomerated2, 'fro')
