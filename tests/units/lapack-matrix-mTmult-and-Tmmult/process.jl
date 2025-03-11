# Real valued
A_real = reshape([sin(i) for i in range(1.0, 15.0)], 3, 5)
B_real = reshape([cos(i) for i in range(1.0, 15.0)], 3, 5)
C1_real = [i * j for i in range(1.0, 3.0), j in range(1.0, 3.0)]
C2_real = [i * j for i in range(1.0, 5.0), j in range(1.0, 5.0)]
alpha_real = 2.5

A_mul_AT_real = A_real * transpose(A_real)
AT_mul_A_real = transpose(A_real) * A_real
A_mul_BT_real = A_real * transpose(B_real)
AT_mul_B_real = transpose(A_real) * B_real

C1_add_A_mul_AT_real = alpha_real * A_real * transpose(A_real) + C1_real
C2_add_AT_mul_A_real = alpha_real * transpose(A_real) * A_real + C2_real
C1_add_A_mul_BT_real = alpha_real * A_real * transpose(B_real) + C1_real
C2_add_AT_mul_B_real = alpha_real * transpose(A_real) * B_real + C2_real

# Complex valued
A_complex = reshape([sin(i) + cos(i)im for i in range(1.0, 15.0)], 3, 5)
B_complex = reshape([cos(i) + sin(i)im for i in range(1.0, 15.0)], 3, 5)
C1_complex = [i*j + (i*j)im for i in range(1.0, 3.0), j in range(1.0, 3.0)]
C2_complex = [i*j + (i*j)im for i in range(1.0, 5.0), j in range(1.0, 5.0)]
alpha_complex = 2.5 + 1.2im

A_mul_AT_complex = A_complex * transpose(A_complex)
AT_mul_A_complex = transpose(A_complex) * A_complex
A_mul_BT_complex = A_complex * transpose(B_complex)
AT_mul_B_complex = transpose(A_complex) * B_complex

C1_add_A_mul_AT_complex = alpha_complex * A_complex * transpose(A_complex) + C1_complex
C2_add_AT_mul_A_complex = alpha_complex * transpose(A_complex) * A_complex + C2_complex
C1_add_A_mul_BT_complex = alpha_complex * A_complex * transpose(B_complex) + C1_complex
C2_add_AT_mul_B_complex = alpha_complex * transpose(A_complex) * B_complex + C2_complex
