/** \file laplace_bem.h
 * \brief Implementation of BEM involving kernel functions and singular
 * numerical quadratures.
 *
 * \ingroup sauter_quadrature
 * \date 2020-11-02
 * \author Jihuan Tian
 */

#ifndef INCLUDE_LAPLACE_BEM_H_
#define INCLUDE_LAPLACE_BEM_H_

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

#include "bem_general.h"
#include "bem_tools.h"
#include "laplace_kernels.h"
#include "quadrature.templates.h"
#include "sauter_quadrature.h"

#endif /* INCLUDE_LAPLACE_BEM_H_ */
