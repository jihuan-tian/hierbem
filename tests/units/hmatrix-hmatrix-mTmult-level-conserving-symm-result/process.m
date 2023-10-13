clear all;
load hmatrix-hmatrix-mTmult-level-conserving-symm-result.output;
HH_full = tril2fullsym(H_full);
M = M1 * M1';
printout_var("norm(HH_full - M, 'fro') / norm(M, 'fro')");
