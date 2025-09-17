# Project Introduction

**What is HierBEM**

HierBEM is a scientific C++ software library that implements the 3D Galerkin boundary element method (BEM), accelerated by hierarchical matrices ($\mathcal{H}$-matrices) to achieve log-linear complexity. HierBEM is built on top of the powerful finite element library deal.II, leveraging its mature infrastructure for mesh handling, finite elements, mappings, linear algebra and data output, etc.

**Long Term Objective**

To provide a suite of high-performance solvers, including:
- BEM solvers for various physical problems, such as electromagnetics, acoustics, elasticity, etc.
- BEM solvers for problems with multiple homogeneous subdomains, using domain decomposition methods (DDM),
- BEM/FEM coupled solvers for addressing inhomogeneous materials and nonlinear effects.

**Technical Features**

* Numerical quadrature
  * Accurate evaluation of singular boundary integrals using Sauter quadrature with CUDA acceleration.
  * Sauter Quadrature on curved surfaces via high-order mappings from deal.II.
* $\mathcal{H}$-matrix algebra
  * Formatted addition, matrix/matrix multiplication, matrix/vector multiplication, LU and Cholesky factorization.
  * TBB parallelization for matrix/vector multiplication, LU and Cholesky factorization.
  * Load balancing for parallel $\mathcal{H}$-matrix assembly and matrix/vector multiplication using sequence partitioning.
* Linear solvers and preconditioners
  * Real and complex valued preconditioned CG and GMRES.
  * $\mathcal{H}$-matrix LU and Cholesky factorization as preconditioners.
  * With operator preconditioning based on the pseudo-differential operator theory, the condition number of the preconditioned system becomes independent of discretization, hence applicable to large scale problems.
* BEM solvers
  * Laplace solvers (real and complex) with Dirichlet, Neumann, and mixed boundary conditions.
