using LinearAlgebra

A = transpose(reshape(1.:36., 6, 6))
A_complex = [sin(i) + cos(j)im for i in range(1.0, 6.0), j in range(1.0, 6.0)]

x = collect(range(1.0, 2.0, 6))
y = x * 1.2
x_complex = collect(range(1.0, 2.0, 6)) + collect(range(3.0, 5.0, 6))im
y_complex = x_complex * (1.1 + 2.3im)

y1 = A*x
y2 = y + A*x
y3 = A_complex * x_complex
y4 = y_complex + A_complex * x_complex
