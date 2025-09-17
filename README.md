English | [中文](README-zh.md)

# What is HierBEM?

HierBEM is a scientific C++ software library that implements the **3D Galerkin boundary element method (BEM)**, accelerated by **hierarchical matrices ($\mathcal{H}$-matrices)** to achieve **log-linear complexity**. HierBEM is built on top of the powerful finite element library **deal.II**, leveraging its mature infrastructure for mesh handling, finite elements, mappings, linear algebra and data output, etc.

The long term objective of HierBEM is to provide a suite of high-performance solvers, including:

- BEM solvers for problems with multiple homogeneous subdomains, using **domain decomposition methods (DDM)**,
- **BEM/FEM coupled solvers** for addressing inhomogeneous materials and nonlinear effects.

Target applications include a wide range of physical problems, such as electromagnetics, acoustics, elasticity, and beyond.

# Why $\mathcal{H}$-Matrix Based Galerkin BEM?

* Compared with the **finite element method (FEM)**, the Galerkin BEM offers several advantages:

  * **Open-domain problems**

    BEM naturally handles unbounded open domains, which are common in electromagnetics and acoustics.

  * **Surface-only discretization**

    Only the boundary surface requires meshing, avoiding the complexity of generating full 3D meshes. This is particularly attractive for models with fine structures or extreme size ratios, where 3D meshing may become the bottleneck — or even fail entirely.

  * **Surface quantities of interest**

    Physical quantities such as electric field strength or mechanical stresses often concentrate on surfaces or interfaces, which are critical in engineering design. Galerkin BEM directly computes these boundary quantities from integral equations, whereas FEM typically approximates them from cell-based values.

* Compared with **collocation and Nyström BEM**, the Galerkin formulation avoids the difficulties in handling singularities at corners and edges.

* Compared with **fast multiple method (FMM)** and **panel clustering**, $\mathcal{H}$-matrix based BEM is a purely algebraic approach: it requires no problem-specific kernel expansions, making it more general and widely applicable across different physical models.

# Features

* **Numerical quadrature**
  * Accurate evaluation of singular boundary integrals using **Sauter quadrature**.
  * **CUDA parallelization** to accelerate computationally intensive quadrature.
  * Support for curved surfaces via **high-order mappings** from deal.II.
* **$\mathcal{H}$-matrix algebra**
  * Formatted addition, matrix/matrix multiplication, and matrix/vector multiplication.
  * **TBB task parallelization** for matrix/vector multiplication.
  * **TBB flow graph parallelization** for LU and Cholesky factorization.
  * **Load balancing** for parallel $\mathcal{H}$-matrix assembly and matrix/vector multiplication using **sequence partitioning**.
* **Linear solvers and preconditioners**
  * Iterative solvers for real and **complex-valued** systems, including **preconditioned CG** and **GMRES**.
  * **$\mathcal{H}$-matrix LU and Cholesky factorization** as preconditioners.
  * **Operator preconditioning** based on the pseudo-differential operator theory:
    * Ensures spectral equivalence between the system and preconditioning matrices.
    * Condition number of the preconditioned system becomes independent of discretization.
    * Enables efficient solution of large-scale BEM problems.
* **BEM solvers**
  * Laplace solvers (real and complex) with Dirichlet, Neumann, and mixed boundary conditions.

# Next steps

* **Helmholtz solver** — Develop a complex valued solver for the Helmholtz equation to address acoustic problems.
* **Multi-domain BEM** — Extend the existing single domain BEM solver to handle multiple homogeneous subdomains using domain decomposition methods (DDM).
* **BEM/FEM coupling** — Integrate HierBEM with the FEM framework in deal.II, combining the strengths of BEM for open and interface problems with FEM's ability to model inhomogeneous and nonlinear domains.

# Dependencies

* [GCC](https://gcc.gnu.org/) ≥ 12.2.0
* [CMake](https://cmake.org/) ≥ 3.25
* [CUDA](https://developer.nvidia.com/cuda-toolkit) ≥ 12.4
* [OpenCASCADE](https://github.com/Open-Cascade-SAS/OCCT) 7.8.0
* [Gmsh 4.14.0](https://github.com/jihuan-tian/gmsh) (modified version)
* [deal.II 9.4.1](https://github.com/jihuan-tian/dealii) (modified version)
* [reflect-cpp](https://github.com/getml/reflect-cpp)
* [GNU Octave](https://octave.org/) ≥ 7.3.0
* [Julia](https://julialang.org/) ≥ 1.10

# Build HierBEM

Please see [BUILD.md](BUILD.md) for details.

# Examples

HierBEM comes with a collection of test cases that demonstrate how to setup and solve boundary integral equations using Galerkin BEM.

* Laplace BEM solvers — Example codes can be found in [hierbem/tests/functions/laplace-bem](tests/functions/laplace-bem) .

For a quick start, here is a minimal example work flow in C++:

1. Outside the `main` function, define a class `DirichletBC` as the boundary condition by inheriting from deal.II's `Function` class. This boundary condition describes the potential generated by a positive unit point charge located at the point `x0`.

   ```c++
   class DirichletBC : public Function<3>
   {
   public:
     // N.B. This function should be defined outside class NeumannBC or class
     // Example2, if no inline.
     DirichletBC()
       : Function<3>()
       , x0(0.25, 0.25, 0.25)
     {}
   
     DirichletBC(const Point<3> &x0)
       : Function<3>()
       , x0(x0)
     {}
   
     double
     value(const Point<3> &p, const unsigned int component = 0) const
     {
       (void)component;
       return 1.0 / 4.0 / numbers::PI / (p - x0).norm();
     }
   
   private:
     Point<3> x0;
   };
   ```

2. Within the `main` function, create an object `bem` of the class `LaplaceBEM`.

   ```c++
   const unsigned int                        dim                 = 2;
   const unsigned int                        spacedim            = 3;
   const bool                                is_interior_problem = true;
   LaplaceBEM<dim, spacedim, double, double> bem(
     1, // fe order for dirichlet space
     0, // fe order for neumann space
     LaplaceBEM<dim, spacedim, double, double>::ProblemType::DirichletBCProblem,
     is_interior_problem,         // is interior problem
     4,                           // n_min for cluster tree
     4,                           // n_min for block cluster tree
     0.8,                         // eta for H-matrix
     5,                           // max rank for H-matrix
     0.01,                        // aca epsilon for H-matrix
     1.0,                         // eta for preconditioner
     2,                           // max rank for preconditioner
     0.1,                         // aca epsilon for preconditioner
     MultithreadInfo::n_threads() // Number of threads used for ACA
   );
   ```

3. Set other properties of the object `bem`, such as project name, preconditioner type and $\mathcal{H}$-matrix/vector multiplication type (used in the iterative solver).

   ```c++
   bem.set_project_name("laplace-dirichlet");
   bem.set_preconditioner_type(PreconditionerType::OperatorPreconditioning);
   bem.set_iterative_solver_vmult_type(IterativeSolverVmultType::TaskParallel);
   ```

4. Create a triangulation for a unit sphere as the problem domain.

   ```c++
   const Point<spacedim>        center(0, 0, 0);
   const double                 radius(1);
   Triangulation<dim, spacedim> tria;
   GridGenerator::hyper_sphere(tria, center, radius);
   tria.refine_global(3);
   ```

5. Define a spherical manifold and assign it with the manifold id 0.

   ```c++
   SphericalManifold<dim, spacedim> *spherical_manifold =
     new SphericalManifold<dim, spacedim>(center);
   bem.get_manifolds()[0] = spherical_manifold;
   ```

6. Define a map from material id to manifold id. By default, the material id is 0 for all cells in a triangulation created by deal.II's `GridGenerator`. Then the above defined spherical manifold will be assigned to all cells in the triangulation.

   ```c++
   bem.get_manifold_description()[0] = 0;
   ```

7. Define a map from manifold id to the polynomial order of Lagrange mapping. Here we use the 2nd order `MappingQ` to approximate the curved surface of the sphere.

   ```c++
   bem.get_manifold_id_to_mapping_order()[0] = 2;
   ```

8. Build a surface-to-volume and volume-to-surface relationship. The argument `{0}` is a list of material ids for all surfaces in the model. Here we only have one material id 0.

   ```c++
   bem.get_subdomain_topology().generate_single_domain_topology_for_dealii_model(
     {0});
   ```

9. Create a Dirichlet boundary condition object and assign it to `bem`.

   ```c++
   DirichletBC dirichlet_bc(Point<spacedim>(1, 1, 1););
   bem.assign_dirichlet_bc(dirichlet_bc);
   ```

10. Run the solver.

    ```c++
    bem.run();
    ```

After building the example code and running its executable on the command line, a file `laplace-dirichlet.vtk` will be produced. It contains the Dirichlet data (potential) and Neumann data (conormal trace of the potential) distributed on the sphere, which can be visualized in ParaView.

# Citation

If you use HierBEM in your research, please cite it as follows:

```bibtex
@misc{hierbem2025,
  author       = {Jihuan Tian, Xiaozhe Wang},
  title        = {HierBEM: A Hierarchical Matrix Based Galerkin Boundary Element Method Library},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/jihuan-tian/hierbem}},
}
```

# License

This project is licensed under the LGPL v3. Please see [LICENSE](LICENSE) for details.

# Contact

* GitHub Issues: https://github.com/jihuan-tian/hierbem/issues
* Email: jihuan_tian@hotmail.com

# References

1. Arndt, Daniel, Wolfgang Bangerth, Denis Davydov, et al. 2021. “The Deal.II Finite Element Library: Design, Features, and Insights.” *Computers & Mathematics with Applications* 81 (January): 407–22.
2. Steinbach, Olaf. 2007. *Numerical Approximation Methods for Elliptic Boundary Value Problems: Finite and Boundary Elements*. Springer Science & Business Media.
3. Hackbusch, Wolfgang. 2015. *Hierarchical Matrices: Algorithms and Analysis*.
4. Bebendorf, Mario. 2008. *Hierarchical Matrices: A Means to Efficiently Solve Elliptic Boundary Value Problems*. Lecture Notes in Computational Science and Engineering 63. Springer.
5. Sauter, Stefan, and Christoph Schwab. 2010. *Boundary Element Methods*. Springer Science & Business Media.
6. Hiptmair, Ralf, and Carolina Urzua-Torres. 2016. *Dual Mesh Operator Preconditioning On 3D Screens: Low-Order Boundary Element Discretization.* Nos. 2016–14. Seminar für Angewandte Mathematik, Eidgenössische Technische Hochschule.
7. Hiptmair, R. 2006. “Operator Preconditioning.” *Computers & Mathematics with Applications* 52 (5): 699–706.
8. Steinbach, Olaf, and Wolfgang L. Wendland. 1998. “The Construction of Some Efficient Preconditioners in the Boundary Element Method.” *Advances in Computational Mathematics* 9 (1–2): 191–216.
9. Steinbach, O., and W. L. Wendland. 1999. “Domain Decomposition and Preconditioning Techniques in Boundary Element Methods.” In *Boundary Element Topics*, edited by Wolfgang L. Wendland. Springer Berlin Heidelberg.
10. Betcke, Timo, Matthew W. Scroggs, and Wojciech Śmigaj. 2020. “Product Algebras for Galerkin Discretisations of Boundary Integral Operators and Their Applications.” *ACM Trans. Math. Softw.* 46 (1): 4:1-4:22.
