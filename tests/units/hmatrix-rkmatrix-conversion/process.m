clear all;

load_packages;

## Plot the partition structure of all the agglomerations.
## N = 177;

## for m = 0:N
##   figure(m + 1);
##   plot_bct_struct(cstrcat("hmat-bct", num2str(m), ".dat"), false);
##   title("Agglomeration of H-matrix");
##   number_str = sprintf("%03d", m);
##   PrintGCF(cstrcat("hmat-bct", number_str, ".png"));
## endfor

load hmatrix-rkmatrix-conversion.output;

norm(M_agglomerated.A * M_agglomerated.B' - M, "fro") / norm(M, "fro")
