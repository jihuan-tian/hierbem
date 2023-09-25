clear all;
load rkmatrix-agglomeration-interwoven-indices-4th-block-rank0.output;

norm(M11 - M11_rk.A * M11_rk.B', 'fro') / norm(M11, 'fro')
norm(M12 - M12_rk.A * M12_rk.B', 'fro') / norm(M12, 'fro')
norm(M21 - M21_rk.A * M21_rk.B', 'fro') / norm(M21, 'fro')
# norm(M22 - M22_rk.A * M22_rk.B', 'fro') / norm(M22, 'fro')
norm(M - M_rk.A * M_rk.B', 'fro') / norm(M, 'fro')
