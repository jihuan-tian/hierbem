using LinearAlgebra

A = transpose(reshape(1.:36., 6, 6))
A_symm = A * transpose(A)
issymmetric(A_symm)
A_tril = tril(A)
A_triu = triu(A)

A_complex = [sin(i) + cos(j)im for i in range(1.0, 6.0), j in range(1.0, 6.0)]
A_complex_symm = A_complex * transpose(A_complex)
issymmetric(A_complex_symm)
A_complex_hermite_symm = A_complex * adjoint(A_complex)
ishermitian(A_complex_hermite_symm)
A_complex_tril = tril(A_complex)
A_complex_triu = triu(A_complex)

x = collect(range(1.0, 2.0, 6))
y = x * 1.2
x_complex = collect(range(1.0, 2.0, 6)) + collect(range(3.0, 5.0, 6))im
y_complex = x_complex * (1.1 + 2.3im)

y1 = A*x
y2 = y + A*x
y3 = A_symm * x
y4 = y + A_symm * x
y5 = A_tril * x
y6 = y + A_tril * x
y7 = A_triu * x
y8 = y + A_triu * x

y9 = A_complex * x_complex
y10 = y_complex + A_complex * x_complex
y11 = A_complex_symm * x_complex
y12 = y_complex + A_complex_symm * x_complex
y13 = A_complex_hermite_symm * x_complex
y14 = y_complex + A_complex_hermite_symm * x_complex
y15 = A_complex_tril * x_complex
y16 = y_complex + A_complex_tril * x_complex
y17 = A_complex_triu * x_complex
y18 = y_complex + A_complex_triu * x_complex
