load "solve-laplace-dirichlet.output";
load "analytical-solution.output";

figure;
hold on;
plot(analytical_solution, "r.");
plot(solution, "bo");
legend("Analytical", "Numerical");
