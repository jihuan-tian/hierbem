# Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version. The full text of the license can be found in the file
# LICENSE at the top level directory of HierBEM.

using LinearAlgebra, SparseArrays

"""
Apply a ``2×2`` matrix to at vector at the specified index.
"""
function apply_givens_2x2(v, Omega, i)
    vi = Omega[1,1] * v[i] + Omega[1,2] * v[i+1]
    vi1 = Omega[2,1] * v[i] + Omega[2,2] * v[i+1]

    # Directly override the i-th and (i+1)-th entries in the input vector.
    v[i] = vi
    v[i+1] = vi1
end

function check_orthogonality(Vm)
    m = size(Vm, 2)
    orthogonality = zeros(eltype(Vm), m, m)
    for i = 1:m
        for j = 1:(i-1)
            orthogonality[i,j] = dot(Vm[:,i], Vm[:,j])
        end
    end

    return norm(orthogonality)
end

"""
    dqgmres!(x, A, b; krylov_dim::Int = min(20, size(A, 2)), ortho_hist_len::Int = krylov_dim, max_iter::Int = size(A, 2), abs_tol::Real = zero(real(eltype(b))), enable_log::Bool = true)

Direct version of GMRES with restart mechanism.

# Arguments
- `x::Vector{Real}`: Before execution, it holds the initial guess `x0`. After execution, it holds the solution vector `x`.
- `A::Matrix{Real}`: Square matrix belong to ``\\mathbb{K}^{n×n}``.
- `b::Vector{Real}`: Right hand side vector of length `n`.
- `P::Function`: Preconditioner implemented as a function `(x)->y`, which internally perform the matrix/vector multiplication ``y=M^{-1}x``.
- `krylov_dim::Integer`: Krylov subspace dimension. The maximum value of `krylov_dim` is `n`.
- `ortho_hist_len::Integer`: Number of historic vectors used for the orthogonalization. When `ortho_hist_len == krylov_dim`, full Gram-Schmidt orthogonalization is performed; when `ortho_hist_len < krylov_dim`, incomplete Gram-Schmidt orthogonalization is performed.
- `max_iter::Integer`: Maximum number of restart iterations.
- `abs_tol::Real`: Absolute tolerance for the residual vector norm.
- `is_left_precondition::Bool`: If the preconditioner is applied from left or right.
- `enable_log::Bool`: If enable logging.
"""
function dqgmres!(x, A, b, P; krylov_dim::Int = min(20, size(A, 2)), ortho_hist_len::Int = krylov_dim, max_iter::Int = size(A, 2), abs_tol::Real = zero(real(eltype(b))), is_left_precondition::Bool = true, enable_log::Bool = true)
    # Vector length.
    n = size(A, 1)

    # Detect the number type used by the right hand side vector.
    VectorNumberType = eltype(b)
    is_real = isreal(b)

    # Matrix holding the orthonormal basis of the Krylov subspace. N.B. One more
    # column vector is appended to ``V_m``, which is orthogonal to the Krylov
    # subspace.
    Vm = zeros(VectorNumberType, (n, krylov_dim + 1))

    # Last column of the Hessenberg matrix ``\\overline{H}_m``.
    hm = zeros(VectorNumberType, krylov_dim + 1)

    Pm = zeros(VectorNumberType, (n, krylov_dim))

    # Assistant vector for error estimate when `ortho_hist_len < krylov_dim`.
    Zm = zeros(VectorNumberType, n)

    # Givens transformation matrices.
    givens_mats = zeros(VectorNumberType, (2, 2, krylov_dim))

    # Initial residual vector
    r0 = zeros(VectorNumberType, n);
    
    residual_norm = 0.0

    # Outer loop for restarting the GMRES.
    restart_index = 1
    while restart_index <= max_iter
        if enable_log
            @info "Restart $restart_index"
        end

        if is_left_precondition
            r0 = P(b - A * x)
        else
            r0 = b - A * x
        end

        residual_norm = norm(r0)
        gamma_current = VectorNumberType(residual_norm)
        if restart_index == 1
            if enable_log
                println("Restart index=0, Residual norm=", residual_norm)
            end

            if residual_norm <= abs_tol
                return residual_norm, 0
            end
        end

        # Clear the matrices and vectors.
        Vm .= VectorNumberType(0.)
        Pm .= VectorNumberType(0.)
        givens_mats .= VectorNumberType(0.)

        Vm[:, 1] = r0 / gamma_current
        # When `ortho_hist_len < krylov_dim`, i.e. incomplete Gram-Schimdt
        # orthogonalization, we need `Zm` to get an accurate error estimate.
        if ortho_hist_len < krylov_dim
            Zm = Vm[:,1]
        end
        
        # Inner loop for increasing the Krylov subspace dimension from 1 to `m`.
        for j = 1:krylov_dim            
            # Perform a full or incomplete Gram-Schmidt orthogonalization.
            if is_left_precondition
                Vm[:,j+1] = P(A * Vm[:,j])
            else
                Vm[:,j+1] = A * P(Vm[:,j])
            end
            
            # Compute the last column of the Hessenberg matrix ``\\overline{H}_j``.
            hm .= VectorNumberType(0.0)
            for i = max(1, j-ortho_hist_len+1):j
                hm[i] = dot(Vm[:,i], Vm[:,j+1])
                Vm[:,j+1] .-= hm[i] * Vm[:,i]
            end

            hm[j+1] = VectorNumberType(norm(Vm[:,j+1]))
            Vm[:,j+1] ./= hm[j+1]

            # Apply historic Givens transformation to the last column of ``\\overline{H}_j``.
            for i = max(1, j-ortho_hist_len):j-1
                apply_givens_2x2(hm, givens_mats[:,:,i], i)
            end

            # Compute the current Givens transformation matrix.
            if is_real
                denominator = sqrt(hm[j]^2 + hm[j+1]^2)
            else
                denominator = sqrt(abs(hm[j])^2 + abs(hm[j+1])^2)
            end
            
            sj = hm[j+1] / denominator
            cj = hm[j] / denominator

            if is_real
                givens_mats[1,1,j] = cj
                givens_mats[1,2,j] = sj
                givens_mats[2,1,j] = -sj
                givens_mats[2,2,j] = cj
            else
                givens_mats[1,1,j] = conj(cj)
                givens_mats[1,2,j] = conj(sj)
                givens_mats[2,1,j] = -sj
                givens_mats[2,2,j] = cj
            end

            # Apply the current Givens transformation to the last column of the
            # Hessenberg matrix and the right hand side vector.
            hm[j] = givens_mats[1,1,j] * hm[j] + givens_mats[1,2,j] * hm[j+1]
            hm[j+1] = VectorNumberType(0.0)
            gamma_next = givens_mats[2,1,j] * gamma_current
            gamma_current = givens_mats[1,1,j] * gamma_current

            # Compute the vector `pj`.
            Pm[:,j] = Vm[:,j]
            for i = max(1,j-ortho_hist_len):j-1
                Pm[:,j] .-= hm[i] * Pm[:,i]
            end
            Pm[:,j] ./= hm[j]

            # Update the solution vector `x`.
            if is_left_precondition
                x .+= gamma_current * Pm[:,j]
            else
                x .+= gamma_current * P(Pm[:,j])
            end
            
            # Compute residual norm.
            if ortho_hist_len < krylov_dim
                Zm = givens_mats[2,1,j] * Zm + givens_mats[2,2,j] * Vm[:,j+1]
                residual_norm = abs(gamma_next) * norm(Zm)
            else
                residual_norm = abs(gamma_next)
            end

            if enable_log
                println("Restart index=", restart_index, ", Iteration=", j, ", Residual norm=", residual_norm)
            end

            # Check convergence.
            if residual_norm <= abs_tol
                return residual_norm, restart_index
            else
                # Update variable values.
                gamma_current = gamma_next
            end
        end

        restart_index += 1
    end
    
    return residual_norm, restart_index - 1
end
