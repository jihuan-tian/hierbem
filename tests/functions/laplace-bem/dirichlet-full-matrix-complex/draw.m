figure;
scale_fig(gcf, [2, 1])
subplot(1, 2, 1);
hold on;
plot(real(solution_ref), 'bo');
plot(real(solution), 'r.');
legend('Reference', 'Solution');
title("Real part");
grid on;
hold off;

subplot(1, 2, 2);
hold on;
plot(imag(solution_ref), 'bo');
plot(imag(solution), 'r.');
legend('Reference', 'Solution');
title("Imaginary part");
grid on;
hold off;
