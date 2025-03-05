using LinearAlgebra

values = [3.0, 8.0, 10.0, 7.0, 1.0, 9.0, 7.0, 6.0, 12.0, 4.0, 5.0, 8.0, 8.0, 9.0, 20.0]
values_c = [3.0 + 2.0im, 8.0 + 1.0im, 10.0 + 3.0im, 7.0 + 5.0im, 1.0 + 0.3im, 9.0 + 7.0im,
            7.0 + 10.0im, 6.0 + 1.0im, 12.0 + 0.8im, 4.0 + 2.0im, 5.0 + 7.0im, 8.0 +
                10.0im, 8.0 + 3.3im, 9.0 + 7.0im, 20.0 + 12.0im]

# QR decomposition of a matrix with more columns than rows.
M1 = reshape(values, 3, 5)
Mc1 = reshape(values_c, 3, 5)
F1 = qr(M1)
Q1 = Matrix(F1.Q)
R1 = F1.R

Fc1 = qr(Mc1)
Qc1 = Matrix(Fc1.Q)
Rc1 = Fc1.R

# QR decomposition of a matrix with more rows than columns. N.B. Julia always
# performs reduced QR decomposition. The result Q and R will be converted to
# non-reduced form to be compared wth C++.
M2 = reshape(values, 5, 3)
Mc2 = reshape(values_c, 5, 3)
F2 = qr(M2)
# The unitary matrix returned in F2.Q is in a compact form, which should be
# converted to a full matrix. We need multiply F2.Q with an identity matrix to
# get its original complete form.
Q2 = F2.Q * I
R2 = zeros(5, 3)
R2[1:3,1:3] = F2.R

Fc2 = qr(Mc2)
Qc2 = Fc2.Q * I
Rc2 = zeros(Complex{Float64}, 5, 3)
Rc2[1:3, 1:3] = Fc2.R

# Now Q and R are in the compact form.
Q3 = Matrix(F2.Q)
R3 = F2.R
Qc3 = Matrix(Fc2.Q)
Rc3 = Fc2.R
