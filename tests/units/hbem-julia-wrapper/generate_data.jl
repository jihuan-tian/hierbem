a = UInt32(10)
b = Int32(-5)
c = Float32(3.5)
d = Float64(-3.5)
e = ComplexF32(1.2 + 0.5im)
f = ComplexF64(-1.2 - 0.5im)
g = collect(Float32, range(1.0, 5.0, step = 0.2))
h = collect(Float64, range(5.0, 1.0, step = -0.2))
i = [ComplexF32(x + (x + 1)im) for x in range(1.0, 5.0, step = 0.2)]
j = [ComplexF64(x + (x + 1)im) for x in range(5.0, 1.0, step = -0.2)]
k = reshape(h, 3, 7)
