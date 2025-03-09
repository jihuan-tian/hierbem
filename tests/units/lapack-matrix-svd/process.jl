using LinearAlgebra

A = transpose(reshape(1.:15., 5, 3))
F = svd(A, full = true)
U = Matrix(F.U)
VT = Matrix(F.Vt)
Sigma_r = F.S

A_complex = [sin(i) + cos(j)im for i in range(1.0, 3.0), j in range(1.0, 5.0)]
F_complex = svd(A_complex, full = true)
U_complex = Matrix(F_complex.U)
VT_complex = Matrix(F_complex.Vt)
Sigma_r_complex = F_complex.S
