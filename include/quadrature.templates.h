/**
 * @file quadrature.templates.h
 * @brief This file is extracted from $DEAL_II_DIR/source/base/quadrature.cc
 * for generating the Gauss quadrature in 4 dimensional space from the template.
 *
 * @date 2020-11-16
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_QUADRATURE_TEMPLATES_H_
#define HIERBEM_INCLUDE_QUADRATURE_TEMPLATES_H_

#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

DEAL_II_NAMESPACE_OPEN

template <int dim>
Quadrature<dim>::Quadrature(const unsigned int n_q)
  : quadrature_points(n_q, Point<dim>())
  , weights(n_q, 0)
  , is_tensor_product_flag(dim == 1)
{}


template <int dim>
void
Quadrature<dim>::initialize(const std::vector<Point<dim>> &p,
                            const std::vector<double>     &w)
{
  AssertDimension(w.size(), p.size());
  quadrature_points      = p;
  weights                = w;
  is_tensor_product_flag = dim == 1;
}


template <int dim>
Quadrature<dim>::Quadrature(const std::vector<Point<dim>> &points,
                            const std::vector<double>     &weights)
  : quadrature_points(points)
  , weights(weights)
  , is_tensor_product_flag(dim == 1)
{
  Assert(weights.size() == points.size(),
         ExcDimensionMismatch(weights.size(), points.size()));
}


template <int dim>
Quadrature<dim>::Quadrature(const std::vector<Point<dim>> &points)
  : quadrature_points(points)
  , weights(points.size(), std::numeric_limits<double>::infinity())
  , is_tensor_product_flag(dim == 1)
{
  Assert(weights.size() == points.size(),
         ExcDimensionMismatch(weights.size(), points.size()));
}


template <int dim>
Quadrature<dim>::Quadrature(const Point<dim> &point)
  : quadrature_points(std::vector<Point<dim>>(1, point))
  , weights(std::vector<double>(1, 1.))
  , is_tensor_product_flag(true)
  , tensor_basis(new std::array<Quadrature<1>, dim>())
{
  for (unsigned int i = 0; i < dim; ++i)
    {
      const std::vector<Point<1>> quad_vec_1d(1, Point<1>(point[i]));
      (*tensor_basis)[i] = Quadrature<1>(quad_vec_1d, weights);
    }
}


template <int dim>
Quadrature<dim>::Quadrature(const SubQuadrature &q1, const Quadrature<1> &q2)
  : quadrature_points(q1.size() * q2.size())
  , weights(q1.size() * q2.size())
  , is_tensor_product_flag(q1.is_tensor_product())
{
  unsigned int present_index = 0;
  for (unsigned int i2 = 0; i2 < q2.size(); ++i2)
    for (unsigned int i1 = 0; i1 < q1.size(); ++i1)
      {
        // compose coordinates of new quadrature point by tensor product in the
        // last component
        for (unsigned int d = 0; d < dim - 1; ++d)
          quadrature_points[present_index](d) = q1.point(i1)(d);
        quadrature_points[present_index](dim - 1) = q2.point(i2)(0);

        weights[present_index] = q1.weight(i1) * q2.weight(i2);

        ++present_index;
      }

#ifdef DEBUG
  if (size() > 0)
    {
      double sum = 0;
      for (unsigned int i = 0; i < size(); ++i)
        sum += weights[i];
      // we cannot guarantee the sum of weights to be exactly one, but it should
      // be near that.
      Assert((sum > 0.999999) && (sum < 1.000001), ExcInternalError());
    }
#endif

  if (is_tensor_product_flag)
    {
      tensor_basis = std::make_unique<std::array<Quadrature<1>, dim>>();
      for (unsigned int i = 0; i < dim - 1; ++i)
        (*tensor_basis)[i] = q1.get_tensor_basis()[i];
      (*tensor_basis)[dim - 1] = q2;
    }
}


template <int dim>
Quadrature<dim>::Quadrature(const Quadrature<dim != 1 ? 1 : 0> &q)
  : Subscriptor()
  , quadrature_points(Utilities::fixed_power<dim>(q.size()))
  , weights(Utilities::fixed_power<dim>(q.size()))
  , is_tensor_product_flag(true)
{
  Assert(dim <= 4, ExcNotImplemented());

  const unsigned int n0 = q.size();
  const unsigned int n1 = (dim > 1) ? n0 : 1;
  const unsigned int n2 = (dim > 2) ? n0 : 1;
  const unsigned int n3 = (dim > 3) ? n0 : 1;

  unsigned int k = 0;

  // N.B. The following embedded loops show that the first coordinate component
  // runs the fastest.
  for (unsigned int i3 = 0; i3 < n3; ++i3)
    for (unsigned int i2 = 0; i2 < n2; ++i2)
      for (unsigned int i1 = 0; i1 < n1; ++i1)
        for (unsigned int i0 = 0; i0 < n0; ++i0)
          {
            quadrature_points[k](0) = q.point(i0)(0);
            if (dim > 1)
              quadrature_points[k](1) = q.point(i1)(0);
            if (dim > 2)
              quadrature_points[k](2) = q.point(i2)(0);
            if (dim > 3)
              quadrature_points[k](3) = q.point(i3)(0);

            weights[k] = q.weight(i0);
            if (dim > 1)
              weights[k] *= q.weight(i1);
            if (dim > 2)
              weights[k] *= q.weight(i2);
            if (dim > 3)
              weights[k] *= q.weight(i3);

            ++k;
          }

  tensor_basis = std::make_unique<std::array<Quadrature<1>, dim>>();
  for (unsigned int i = 0; i < dim; ++i)
    (*tensor_basis)[i] = q;
}


template <int dim>
Quadrature<dim>::Quadrature(const Quadrature<dim> &q)
  : Subscriptor()
  , quadrature_points(q.quadrature_points)
  , weights(q.weights)
  , is_tensor_product_flag(q.is_tensor_product_flag)
{
  if (dim > 1 && is_tensor_product_flag)
    tensor_basis =
      std::make_unique<std::array<Quadrature<1>, dim>>(*q.tensor_basis);
}


template <int dim>
Quadrature<dim> &
Quadrature<dim>::operator=(const Quadrature<dim> &q)
{
  weights                = q.weights;
  quadrature_points      = q.quadrature_points;
  is_tensor_product_flag = q.is_tensor_product_flag;
  if (dim > 1 && is_tensor_product_flag)
    {
      if (tensor_basis == nullptr)
        tensor_basis =
          std::make_unique<std::array<Quadrature<1>, dim>>(*q.tensor_basis);
      else
        *tensor_basis = *q.tensor_basis;
    }
  return *this;
}


template <int dim>
bool
Quadrature<dim>::operator==(const Quadrature<dim> &q) const
{
  return ((quadrature_points == q.quadrature_points) && (weights == q.weights));
}


template <int dim>
std::size_t
Quadrature<dim>::memory_consumption() const
{
  return (MemoryConsumption::memory_consumption(quadrature_points) +
          MemoryConsumption::memory_consumption(weights));
}


/**
 * This is a template version of the constructor for @p QGauss.
 *
 * \mycomment{In @p quadrature_lib.cc, this template constructor exists.
 * However, it is not visible to the user code. Therefore, it is copied here.}
 *
 * @param n
 */
template <int dim>
QGauss<dim>::QGauss(const unsigned int n)
  : Quadrature<dim>(QGauss<dim - 1>(n), QGauss<1>(n))
{}


DEAL_II_NAMESPACE_CLOSE


#endif // HIERBEM_INCLUDE_QUADRATURE_TEMPLATES_H_
