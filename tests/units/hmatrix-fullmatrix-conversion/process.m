clear all;
format long;
load "hmatrix-fullmatrix-conversion.output";

N = 10;
relative_errors = zeros(N, 1);
for m = 1:N
  var_name = cstrcat("M_tilde", num2str(m));
  relative_errors(m) = eval(cstrcat("norm(M - ", var_name, ", 'fro') / norm(M, 'fro')"))
endfor

bar(relative_errors);
xlabel("Rank")
ylabel("Relative Frobenius errors");

## Plot the bct structure with ranks.
figure;
plot_bct_struct("bct-struct-with-rank.dat");
