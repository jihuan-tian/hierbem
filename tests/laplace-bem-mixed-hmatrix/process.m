clear all;
load matrices.dat;
load laplace-bem-mixed-hmatrix.output;

## Restore V1 and D1 to symmetric matrices.
V1_symm = tril2fullsym(V1);
D1_symm = tril2fullsym(D1);

M = [V1_symm, K1; -K1', D1_symm];
solution_manual = M \ system_rhs_on_combined_domain;
printout_var("norm(solution_manual - solution_on_combined_domain_internal_dof_numbering, 2) / norm(solution_on_combined_domain_internal_dof_numbering, 2)");
printout_var("norm(solution_manual - solution_on_combined_domain_internal_dof_numbering, 2)");

## Check the consistency between the manually calculated solution vector and the
## C++ results.
figure;
plot(solution_on_combined_domain_internal_dof_numbering, 'ro', "markersize", 10);
hold on;
plot(solution_manual, 'b+', "markersize", 10);
plot(solution_on_combined_domain_internal_dof_numbering(175:end), 'gx', "markersize", 20);

## Check the consistency between analytical solution on Dirichlet and Neumann
## domains and the numerical solution.
figure;
hold on;
plot(analytical_solution_on_dirichlet_domain, "ro", "markersize", 10);
plot(solution_on_dirichlet_domain, "b+", "markersize", 10);

figure;
hold on;
plot(analytical_solution_on_neumann_domain, "ro", "markersize", 10);
plot(solution_on_neumann_domain, "b+", "markersize", 10);

## Check the consistency between analytical solution on Dirichlet and Neumann
## domains and the numerical solution in the internal numbering.
solution_on_dirichlet_domain_internal_numbering = solution_on_combined_domain_internal_dof_numbering(1:size(V1, 1));
solution_on_neumann_domain_internal_numbering = solution_on_combined_domain_internal_dof_numbering(size(V1, 1)+1:end);

figure;
hold on;
plot(analytical_solution_on_dirichlet_domain_internal_numbering, "ro", "markersize", 10);
plot(solution_on_dirichlet_domain_internal_numbering, "b+", 'markersize', 10);

figure;
hold on;
plot(analytical_solution_on_neumann_domain_internal_numbering, "ro", "markersize", 10);
plot(solution_on_neumann_domain_internal_numbering, "b+", 'markersize', 10);

## Substitue the analytical solution in the internal numbering into the system
## equation and check its error with the RHS vector.
analytical_solution_combined_internal_numbering = [analytical_solution_on_dirichlet_domain_internal_numbering; analytical_solution_on_neumann_domain_internal_numbering];
printout_var("norm(M * analytical_solution_combined_internal_numbering - system_rhs_on_combined_domain, 2) / norm(system_rhs_on_combined_domain, 2)")

## Check the calculation of RHS vector: correct.
y1 = K2 * dirichlet_bc_internal_dof_numbering + V2 * neumann_bc_internal_dof_numbering;
y2 = D2 * dirichlet_bc_internal_dof_numbering + K_prime2 * neumann_bc_internal_dof_numbering;

figure;
hold on;
plot(system_rhs_on_combined_domain, 'ro', 'markersize', 10);
plot([y1;y2], 'b+', 'markersize', 10);
