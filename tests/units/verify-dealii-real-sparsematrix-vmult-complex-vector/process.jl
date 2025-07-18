using LinearAlgebra, SparseArrays

rows = Int32[1,3,4,2,1,3,1,4,1,5]
cols = Int32[1,1,1,2,3,3,4,4,5,5]
vals = [5.,-2.,-4.,5.,-3.,-1.,-2.,-10.,7.,9]

A = sparse(rows, cols, vals, 5, 5)
x = collect(range(1.0, 2.0, 5)) + collect(range(3.0, 5.0, 5))im
y = A * x
