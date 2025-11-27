load "solve-laplace-neumann.output";
load "analytical-solution.output";

## Compute the average shift between the numerical solution and the analytical
## solution.
average_shift = average(analytical_solution - solution);

figure;
scale_fig(gcf, 2);
hold on;
plot(analytical_solution, "r.");
plot(solution + average_shift, "bo");
legend("Analytical", "Numerical", 'location', 'southoutside');
