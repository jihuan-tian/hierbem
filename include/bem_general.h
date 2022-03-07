/**
 * @file bem_general.h
 * @brief Introduction of bem_general.h
 *
 * @date 2022-03-04
 * @author Jihuan Tian
 */
#ifndef INCLUDE_BEM_GENERAL_H_
#define INCLUDE_BEM_GENERAL_H_

#include <deal.II/base/point.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.templates.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "bem_tools.h"
#include "quadrature.templates.h"
#include "sauter_quadrature.h"


namespace IdeoBEM
{
  using namespace dealii;
  using namespace BEMTools;


} // namespace IdeoBEM


#endif /* INCLUDE_BEM_GENERAL_H_ */
