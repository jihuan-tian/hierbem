using LinearAlgebra

# Vector addition
v1 = [a + a * 1.5im for a in 0:9]
v2 = [sin(a) + cos(a)im for a in 0:9]
v3 = v1 + (1.0+0.5im)*v2

# Scalar or inner product
v1 = [sin(a+1) + cos(a+1)im for a in 0:9]
v2 = [(a+1) + (a+1.5)im for a in 0:9]
# N.B. In Julia, complex conjugation is applied to the first operand, while in deal.ii it is applied to the second operand.
v1_dot_v2 = dot(v2, v1)

# Addition then scalar product
v1 = v1 + (1.0+0.5im) * v2
v4 = [tan(a+1) + sqrt(a+1)im for a in 0:9]
add_and_dot = dot(v4, v1)
