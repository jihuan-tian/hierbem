using LinearAlgebra

A = transpose(reshape(1.:36., 6, 6))
A_symm = A * transpose(A)
issymmetric(A_symm)

A_complex = [sin(i) + cos(j)im for i in range(1.0, 6.0), j in range(1.0, 6.0)]
A_complex_symm = A_complex * transpose(A_complex)
issymmetric(A_complex_symm)
