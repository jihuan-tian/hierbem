// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>

#include <boost/math/constants/constants.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>

#include "bem/bem_general.hcu"
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "laplace/laplace_bem.h"
#include "platform_shared/laplace_kernels.h"
#include "postprocessing/data_out_ext.h"
#include "preconditioners/preconditioner_type.h"
#include "utilities/cu_profile.hcu"
#include "utilities/debug_tools.h"

using namespace dealii;
using namespace HierBEM;

class DirichletBC : public Function<3, std::complex<double>>
{
public:
  std::complex<double>
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    const double angle = numbers::PI / 3.0;
    if (p(0) < 0)
      {
        return std::complex<double>(10 * std::cos(angle), 10 * std::sin(angle));
      }
    else
      {
        return std::complex<double>(-10 * std::cos(angle),
                                    -10 * std::sin(angle));
      }
  }
};


namespace HierBEM
{
  namespace CUDAWrappers
  {
    extern cudaDeviceProp device_properties;
  }
} // namespace HierBEM

/**
 * Output the results of potential and conormal trace at a plane and results
 * of potential for a volume.
 */
void
output_results_at_targets(LaplaceBEM<2, 3, std::complex<double>, double> &bem)
{
  const auto &fe_dirichlet = bem.get_dof_handler_dirichlet().get_fe();
  const auto &fe_neumann   = bem.get_dof_handler_neumann().get_fe();
  const Vector<std::complex<double>> &dirichlet_data = bem.get_dirichlet_data();
  const Vector<std::complex<double>> &neumann_data   = bem.get_neumann_data();
  const auto &dof_handler_dirichlet = bem.get_dof_handler_dirichlet();
  const auto &dof_handler_neumann   = bem.get_dof_handler_neumann();

  PlatformShared::LaplaceKernel::SingleLayerKernel<3, double>        V;
  PlatformShared::LaplaceKernel::DoubleLayerKernel<3, double>        K;
  PlatformShared::LaplaceKernel::AdjointDoubleLayerKernel<3, double> K_prime;
  PlatformShared::LaplaceKernel::HyperSingularKernel<3, double>      D;

  {
    Triangulation<2, 3> plane;
    GridGenerator::subdivided_hyper_rectangle(plane,
                                              {50, 50},
                                              Point<2>(-4, -4),
                                              Point<2>(4, 4));
    GridTools::rotate(boost::math::constants::pi<double>() / 2, 1, plane);
    GridOut().write_msh(plane, "plane.msh");

    DoFHandler<2, 3> dof_handler_potential(plane);
    DoFHandler<2, 3> dof_handler_conormal_trace(plane);
    dof_handler_potential.distribute_dofs(fe_dirichlet);
    dof_handler_conormal_trace.distribute_dofs(fe_neumann);

    Vector<std::complex<double>> potentials(dof_handler_potential.n_dofs());
    Vector<std::complex<double>> conormal_traces(
      dof_handler_conormal_trace.n_dofs());

    evaluate_representation_formula_for_potential(V,
                                                  K,
                                                  dof_handler_dirichlet,
                                                  dof_handler_neumann,
                                                  dirichlet_data,
                                                  neumann_data,
                                                  dof_handler_potential,
                                                  1,
                                                  potentials,
                                                  false);

    evaluate_representation_formula_for_conormal_trace(
      K_prime,
      D,
      dof_handler_dirichlet,
      dof_handler_neumann,
      dirichlet_data,
      neumann_data,
      dof_handler_conormal_trace,
      1,
      conormal_traces,
      false);

    std::ofstream                           vtk_result("plane.vtk");
    DataOut<2, 3>                           data_out;
    ComplexOutputDataVector<Vector, double> potentials_vector(potentials);
    ComplexOutputDataVector<Vector, double> conormal_traces_vector(
      conormal_traces);
    add_complex_data_vector(data_out,
                            dof_handler_potential,
                            potentials_vector,
                            "potential");
    add_complex_data_vector(data_out,
                            dof_handler_conormal_trace,
                            conormal_traces_vector,
                            "conormal_traces");
    data_out.build_patches();
    data_out.write_vtk(vtk_result);
  }

  {
    Triangulation<3, 3> cube;
    GridGenerator::subdivided_hyper_cube(cube, 30, 3., 6.);
    GridOut().write_msh(cube, "cube.msh");

    // On target volume, the original surface finite elements in the bem solver
    // cannot be used, so we create them here.
    FE_Q<3, 3>       fe_dirichlet(1);
    FE_DGQ<3, 3>     fe_neumann(0);
    DoFHandler<3, 3> dof_handler_potential(cube);
    DoFHandler<3, 3> dof_handler_conormal_trace(cube);
    dof_handler_potential.distribute_dofs(fe_dirichlet);
    dof_handler_conormal_trace.distribute_dofs(fe_neumann);

    Vector<std::complex<double>> potentials(dof_handler_potential.n_dofs());
    Vector<std::complex<double>> conormal_traces(
      dof_handler_conormal_trace.n_dofs());

    evaluate_representation_formula_for_potential(V,
                                                  K,
                                                  dof_handler_dirichlet,
                                                  dof_handler_neumann,
                                                  dirichlet_data,
                                                  neumann_data,
                                                  dof_handler_potential,
                                                  1,
                                                  potentials,
                                                  false);

    std::ofstream                           vtk_result("cube.vtk");
    DataOut<3, 3>                           data_out;
    ComplexOutputDataVector<Vector, double> potentials_vector(potentials);
    add_complex_data_vector(data_out,
                            dof_handler_potential,
                            potentials_vector,
                            "potential");
    data_out.build_patches();
    data_out.write_vtk(vtk_result);
  }
}

void
run_dirichlet_hmatrix_two_spheres_op_precond_complex(
  const IterativeSolverVmultType vmult_type)
{
  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs(
    std::string("dirichlet-hmatrix-two-spheres-op-precond-complex-vmult-") +
    std::string(vmult_type_name(vmult_type)) + std::string(".log"));
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  LogStream::Prefix prefix_string("HierBEM");
#if ENABLE_NVTX == 1
  HierBEM::CUDAWrappers::NVTXRange nvtx_range("HierBEM");
#endif

  /**
   * @internal Create and start the timer.
   */
  Timer timer;

  /**
   * @internal Initialize the CUDA device parameters.
   */
  //  cudaError_t error_code = cudaSetDevice(0);
  //  error_code =
  //    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync);
  //  AssertCuda(error_code);

  const size_t stack_size = 1024 * 10;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  deallog << "CUDA stack size has been set to " << stack_size << std::endl;

  /**
   * @internal Get GPU device properties.
   */
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));

  /**
   * @internal Use 8-byte bank size in shared memory, since double value type is
   * used.
   */
  // AssertCuda(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const bool is_interior_problem = false;
  LaplaceBEM<dim, spacedim, std::complex<double>, double> bem(
    1, // fe order for dirichlet space
    0, // fe order for neumann space
    LaplaceBEM<dim, spacedim, std::complex<double>, double>::ProblemType::
      DirichletBCProblem,
    is_interior_problem,         // is interior problem
    16,                          // n_min for cluster tree
    16,                          // n_min for block cluster tree
    0.8,                         // eta for H-matrix
    10,                          // max rank for H-matrix
    0.01,                        // aca epsilon for H-matrix
    1.0,                         // eta for preconditioner
    5,                           // max rank for preconditioner
    0.1,                         // aca epsilon for preconditioner
    MultithreadInfo::n_threads() // Number of threads used for ACA
  );
  bem.set_project_name("dirichlet-hmatrix-two-spheres-op-precond-complex");
  bem.set_preconditioner_type(PreconditionerType::OperatorPreconditioning);
  bem.set_iterative_solver_vmult_type(vmult_type);

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  std::ifstream mesh_in(HBEM_TEST_MODEL_DIR "two-spheres.msh");
  read_msh(mesh_in, bem.get_triangulation());
  bem.get_subdomain_topology().generate_topology(HBEM_TEST_MODEL_DIR
                                                 "two-spheres.brep",
                                                 HBEM_TEST_MODEL_DIR
                                                 "two-spheres.msh");

  // Generate two sphere manifolds.
  double                   inter_distance = 8.0;
  Manifold<dim, spacedim> *left_sphere_manifold =
    new SphericalManifold<dim, spacedim>(
      Point<spacedim>(-inter_distance / 2.0, 0, 0));
  Manifold<dim, spacedim> *right_sphere_manifold =
    new SphericalManifold<dim, spacedim>(
      Point<spacedim>(inter_distance / 2.0, 0, 0));
  bem.get_manifolds()[0] = left_sphere_manifold;
  bem.get_manifolds()[1] = right_sphere_manifold;

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = 1;
  bem.get_manifold_id_to_mapping_order()[1] = 1;

  // Assign manifolds to surface entities.
  bem.get_manifold_description()[1] = 0;
  bem.get_manifold_description()[2] = 1;

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  // Assign constant Dirichlet boundary conditions.
  DirichletBC dirichlet_bc;
  bem.assign_dirichlet_bc(dirichlet_bc);

  timer.stop();
  print_wall_time(deallog, timer, "assign boundary conditions");

  if (bem.validate_subdomain_topology())
    {
      timer.start();

      bem.run();
      output_results_at_targets(bem);

      timer.stop();
      print_wall_time(deallog, timer, "run the solver");

      deallog << "Program exits with a total wall time " << timer.wall_time()
              << "s" << std::endl;

      bem.print_memory_consumption_table(deallog.get_file_stream());
    }
  else
    {
      deallog << "Invalid subdomains!" << std::endl;
    }

  ofs.close();
}
