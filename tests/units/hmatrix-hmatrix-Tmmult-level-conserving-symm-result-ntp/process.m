clear all;
load hmatrix-hmatrix-Tmmult-level-conserving-symm-result-ntp.output;
HH_full = tril2fullsym(H_full);
M = M1' * M1;
printout_var("norm(HH_full - M, 'fro') / norm(M, 'fro')");
