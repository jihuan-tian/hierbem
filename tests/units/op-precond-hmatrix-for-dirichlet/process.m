load "op-precond-hmatrix-for-dirichlet.log";

size(Cd, 2) == size(Br, 1)
size(Cd, 2) == size(Mr, 1)
size(Mr, 2) == size(Cp, 2)

## Mass matrix: [dual space on dual mesh] x [primal space on primal mesh]
Mdp = Cd * Mr *Cp';
## Galerkin matrix for the preconditioner: [dual space on dual mesh] x [dual
## space on dual mesh]
Bd = Cd * Br * Cd';
## Inverse of the preconditioning matrix: [primal space on primal mesh] x
## [primal space on primal mesh].
Cinv = inv(Mdp) * Bd * inv(Mdp');

figure;
show_matrix(Mdp);
title("Mass matrix\n[dual space on dual mesh]x[primal space on primal mesh]")
figure;
show_matrix(Cinv);
title("Inverse of the preconditioning matrix\n[primal space on primal mesh]x[primal space on primal mesh]");
