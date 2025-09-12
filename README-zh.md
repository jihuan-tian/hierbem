[English]((README.md)) | 中文

# 什么是 HierBEM ？

HierBEM 是一个以三维伽辽金边界元算法为核心的科学计算软件库。 HierBEM 采用层级矩阵 ($\mathcal{H}$-matrices) 加速，实现了对数线性复杂度的边界元算法。 HierBEM 基于强大的有限元库 deal.II 构建，充分利用了 deal.II 提供的有限元基础数据结构与功能函数，包括网格处理、有限元、单元映射、线性代数、数据输出等等。

HierBEM 目前已实现拉普拉斯方程（实值和复值）在狄利克雷、诺依曼以及混合边界条件下的求解。 HierBEM 的终极目标是提供一套高性能求解器，包括：1、使用区域分解法解决包含多个均质子域问题的纯边界元求解器；2、处理非均质材料和非线性问题的边界元/有限元耦合求解器。应用领域包扩电磁学、声学、弹性力学等。

# 为什么采用 $\mathcal{H}$-矩阵伽辽金边界元法？

* 与有限元法相比，伽辽金边界元法具有如下优势：

  * **开域问题**

    边界元法天然地能够严格处理无界开域问题，这在电磁学和声学仿真中很常见。

  * **仅需表面网格**

    只需对实体边界表面进行二维网格划分，而无需生成完整的三维网格。这对于模拟具有细微结构或极端尺寸比例的模型尤为关键：因为在这种情况下，三维网格划分通常是整个仿真计算的瓶颈，甚至会出现划分失败的情况。

  * **关注表面物理量的分布**

    电场强度或机械应力等物理场量通常集中分布于介质表面或不同介质的交界面。这些区域是工程设计中需要加以关注的薄弱点。伽辽金边界元通过求解边界积分方程，可以直接获得表面与交界面处的精确场量分布，而有限元通常只能通过体单元的计算结果近似求得表面结果。

* 与点配法 (collocation) 和 Nyström 边界元法相比，采用伽辽金形式的边界元能够避免处理棱角或边界奇异点的困难。

* 与快速多极子方法 (FMM) 和面板聚类 (panel clustering) 法相比，基于 $\mathcal{H}$-矩阵的边界元算法是一种纯代数方法，无需针对每一类物理方程构建核函数的特定展开形式，因此具有更强的通用性。

# 功能特性

* 数值积分

  * 采用 Sauter 数值积分法精确计算奇异边界积分。
  * 采用 CUDA 并行化提升 Sauter 数值积分的计算速度。
  * 基于 deal.II 提供的高阶映射，支持曲面上的 Sauter 数值积分。

* $\mathcal{H}$-矩阵代数

  * 格式化加法 (formatted addition) 、矩阵/矩阵乘法、矩阵/向量乘法。
  * 采用 TBB 任务并行化加速矩阵/向量乘法。
  * 采用 TBB 流图并行化加速 LU 与科列斯基 (Cholesky) 分解。
  * 基于序列划分 (sequence partitioning) ，实现了并行 $\mathcal{H}$-矩阵组装和矩阵/向量乘法的负载均衡。

* 线性代数方程组求解器和预处理器

  * 实值和复值迭代求解器，包括预处理共轭梯度法 (Conjugate gradient, CG) 与广义极小剩余法 (GMRES) 。
  * 基于 $\mathcal{H}$-矩阵 LU 与科列斯基分解的预处理器。
  * 基于拟微分算子理论的算子预处理器：
    * 保证系统矩阵和预处理矩阵的谱等价性；
    * 预处理后的系统条件数与离散方式无关；
    * 能够高效求解大规模边界元问题。

* 边界元求解器
  * 实值与复值拉普拉斯方程求解器，可处理狄利克雷、诺依曼与混合边界条件。

# 下一步计划

* 开发用于声学问题的亥姆霍兹 (Helmholtz) 方程复数求解器。
* 结合区域分解法，开发支持包含多个均质子域的边界元方法。
* 耦合 HierBEM 中的边界元求解器与 deal.II 中的有限元求解器，处理非均质和非线性问题。

# 依赖的软件包

* [GCC](https://gcc.gnu.org/) ≥ 12.2.0
* [CMake](https://cmake.org/) ≥ 3.25
* [CUDA](https://developer.nvidia.com/cuda-toolkit) ≥ 12.4
* [OpenCASCADE](https://github.com/Open-Cascade-SAS/OCCT) 7.8.0
* [Gmsh 4.14.0](https://github.com/jihuan-tian/gmsh) (modified version)
* [deal.II 9.4.1](https://github.com/jihuan-tian/dealii) (modified version)
* [reflect-cpp](https://github.com/getml/reflect-cpp)
* [GNU Octave](https://octave.org/) ≥ 7.3.0
* [Julia](https://julialang.org/) ≥ 1.10

# 构建 HierBEM

更多细节请参考 [BUILD.md](BUILD.md) 。

# 范例

HierBEM 提供了一组测试用例，展示了如何采用边界元方法设置与求解边界积分方程。包括：

* 拉普拉斯方程边界元求解器 — 示例代码可在这里找到： [hierbem/tests/functions/laplace-bem](tests/functions/laplace-bem) 。

为了方便用户或开发者快速上手，以下给出了一个最小样例流程，展示 C++ 调用 HierBEM 中的相关功能求解狄利克雷边界条件下的拉普拉斯方程：

1. 在 `main` 函数外，定义一个 `DirichletBC` 类作为边界条件。该类继承自 deal.II 的 `Function` 类。该边界条件描述的是由位于 `x0` 处的单位正电荷在三维空间中产生的电位。

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

2. 在 `main` 函数内部，创建一个 `LaplaceBEM` 类的对象 `bem` 。

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

3. 设置 `bem` 对象的属性，包括项目名、预处理器类型、 $\mathcal{H}$-矩阵/向量乘法的类型。该乘法会被迭代求解器调用。

   ```c++
   bem.set_project_name("laplace-dirichlet");
   bem.set_preconditioner_type(PreconditionerType::OperatorPreconditioning);
   bem.set_iterative_solver_vmult_type(IterativeSolverVmultType::TaskParallel);
   ```

4. 创建一个单位球面网格作为问题的求解域。

   ```c++
   const Point<spacedim>        center(0, 0, 0);
   const double                 radius(1);
   Triangulation<dim, spacedim> tria;
   GridGenerator::hyper_sphere(tria, center, radius);
   tria.refine_global(3);
   ```

5. 定义一个球面流形，并为其赋予流形编号 0 。

   ```c++
   SphericalManifold<dim, spacedim> *spherical_manifold =
     new SphericalManifold<dim, spacedim>(center);
   bem.get_manifolds()[0] = spherical_manifold;
   ```

6. 定义一个从材料编号至流形编号的映射。默认情况下，在由 deal.II 的 `GridGenerator` 类生成的网格中，所有单元的材料编号均为 0 。 然后，将上述球面流形的编号 0 赋予网格中的所有单元。

   ```c++
   bem.get_manifold_description()[0] = 0;
   ```

7. 定义一个从流形编号至单元映射多项式阶数的映射。这里我们使用二阶拉格朗日多项式来近球体的弯曲表面。

   ```c++
   bem.get_manifold_id_to_mapping_order()[0] = 2;
   ```

8. 构建模型中的表面至三维实体、三维实体至表面的关联关系。在如下代码中，函数参数 `{0}` 是一组对应于每个表面实体的材料编号。由于我们的模型中只有一个表面，所以该列表中只有一个元素 0 。

   ```c++
   bem.get_subdomain_topology().generate_single_domain_topology_for_dealii_model(
     {0});
   ```

9. 创建狄利克雷边界条件对象，并将其赋予 `bem` 对象。

   ```c++
   DirichletBC dirichlet_bc(Point<spacedim>(1, 1, 1););
   bem.assign_dirichlet_bc(dirichlet_bc);
   ```

10. 运行求解器。

    ```c++
    bem.run();
    ```

构建上述样例代码，并在命令行中运行生成的可执行文件，我们可以得到一个 `laplace-dirichlet.vtk` 文件。其中包含了球面上的狄利克雷数据（位势分布）与诺依曼数据（位势函数的法向导数分布）。该文件可以在 ParaView 中打开并绘图。

# 文献引用

如果您在研究工作中用到了 HierBEM ，请以如下方式引用本项目：

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

# 软件许可证

本项目采用了 LGPL v3 软件许可证，更多细节请参考 [LICENSE](LICENSE) 。

# 联系方式

* GitHub Issues: https://github.com/jihuan-tian/hierbem/issues
* Email: jihuan_tian@hotmail.com

# 参考文献

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
