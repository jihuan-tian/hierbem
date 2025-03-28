using LinearAlgebra

A = transpose(reshape(1.:36., 6, 6))
A_symm = A * transpose(A)
issymmetric(A_symm)
A_tril = tril(A)
A_triu = triu(A)

A_complex = [sin(i) + cos(j)im for i in range(1.0, 6.0), j in range(1.0, 6.0)]
A_complex_symm = A_complex * transpose(A_complex)
issymmetric(A_complex_symm)
A_complex_tril = tril(A_complex)
A_complex_triu = triu(A_complex)

factor = 0.3
factor_complex = 0.3 + 0.7im
