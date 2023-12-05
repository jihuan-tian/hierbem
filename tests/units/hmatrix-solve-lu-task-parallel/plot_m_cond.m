## Plot the condition numbers of the generated random matrices.

total_trials = 10;
cond_numbers = zeros(total_trials, 1);
for m = 0:(total_trials - 1)
  load(cstrcat("M", num2str(m), ".dat"));
  cond_numbers(m+1) = cond(M);
endfor

figure;
set_fig_size(gcf, [2000, 600]);
bar(0:(total_trials - 1), cond_numbers);
set(gca, 'xtick', 0:(total_trials - 1));
xlim([-1, total_trials]);
grid on;
title("Condition numbers of generated random matrices");
