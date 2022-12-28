/**
 * @file mapping_q_generic_ext.h
 * @brief Extend the class @p MappingQGeneric
 *
 * @date 2022-07-08
 * @author Jihuan Tian
 */
#ifndef INCLUDE_MAPPING_Q_GENERIC_EXT_H_
#define INCLUDE_MAPPING_Q_GENERIC_EXT_H_

#include <deal.II/base/point.h>
#include <deal.II/base/tensor_product_polynomials.h>

#include <deal.II/fe/fe_data.h>
#include <deal.II/fe/fe_q_base.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/tria.h>

#include <vector>

using namespace dealii;

template <int dim, int spacedim = dim>
class MappingQGenericExt : public MappingQGeneric<dim, spacedim>
{
public:
  /**
   * Default constructor
   */
  MappingQGenericExt();

  MappingQGenericExt(const unsigned int polynomial_degree);

  /**
   * Copy constructor
   *
   * @param mapping
   */
  MappingQGenericExt(const MappingQGenericExt<dim, spacedim> &mapping);

  /**
   * Compute a list of support points in the real cell in the hierarchic order.
   *
   * \alert{Because this function has non-covariant return type and different
   * cv-qualifier (non-const here) from the function with the same name in the
   * parent class, it overloads but not overrides the function in the parent
   * class. So here we explicitly declare the function in the parent class via
   * @p using, which eliminates the compiler warning @p -Woverloaded-virtual.}
   *
   * @param cell
   */
  using MappingQGeneric<dim, spacedim>::compute_mapping_support_points;
  void
  compute_mapping_support_points(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell);

  /**
   * Transform the point in the unit to the real cell, assuming that the
   * @p support_points have already been computed using the function
   * @p MappingQGenericExt<dim, spacedim>::compute_mapping_support_points.
   *
   * \alert{Because this function has non-covariant return type and different
   * cv-qualifier (non-const here) from the function with the same name in the
   * parent class, it overloads but not overrides the function in the parent
   * class. So here we explicitly declare the function in the parent class via
   * @p using, which eliminates the compiler warning @p -Woverloaded-virtual.}
   *
   * @param p
   * @return
   */
  using MappingQGeneric<dim, spacedim>::transform_unit_to_real_cell;
  Point<spacedim>
  transform_unit_to_real_cell(const Point<dim> &p) const;

  /**
   * Get the const reference to the list of real support points.
   * @return
   */
  const std::vector<Point<spacedim>> &
  get_support_points() const;

  /**
   * Get the mutable reference to the list of real support points.
   * @return
   */
  std::vector<Point<spacedim>> &
  get_support_points();

  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
  get_data(const UpdateFlags      update_flags,
           const Quadrature<dim> &quadrature) const;

private:
  /**
   * List of support points in the hierarchic ordering of the current real cell.
   */
  std::vector<Point<spacedim>> support_points;
};


template <int dim, int spacedim>
MappingQGenericExt<dim, spacedim>::MappingQGenericExt()
  : MappingQGeneric<dim, spacedim>(1)
  , support_points(0)
{}


template <int dim, int spacedim>
MappingQGenericExt<dim, spacedim>::MappingQGenericExt(
  const unsigned int polynomial_degree)
  : MappingQGeneric<dim, spacedim>(polynomial_degree)
  , support_points(0)
{}


template <int dim, int spacedim>
MappingQGenericExt<dim, spacedim>::MappingQGenericExt(
  const MappingQGenericExt<dim, spacedim> &mapping)
  : MappingQGeneric<dim, spacedim>(mapping)
  , support_points(mapping.support_points)
{}


template <int dim, int spacedim>
void
MappingQGenericExt<dim, spacedim>::compute_mapping_support_points(
  const typename Triangulation<dim, spacedim>::cell_iterator &cell)
{
  support_points =
    MappingQGeneric<dim, spacedim>::compute_mapping_support_points(cell);
}


template <int dim, int spacedim>
Point<spacedim>
MappingQGenericExt<dim, spacedim>::transform_unit_to_real_cell(
  const Point<dim> &p) const
{
  // Set up the polynomial space in the lexicographic order.
  const TensorProductPolynomials<dim> tensor_pols(
    Polynomials::generate_complete_Lagrange_basis(
      this->line_support_points.get_points()));
  Assert(tensor_pols.n() ==
           Utilities::fixed_power<dim>(this->polynomial_degree + 1),
         ExcInternalError());

  // Get the numbering for accessing the support points in the lexicographic
  // ordering which are stored in the hierarchic ordering.
  const std::vector<unsigned int> renumber(
    FETools::lexicographic_to_hierarchic_numbering(this->polynomial_degree));

  Point<spacedim> mapped_point;
  for (unsigned int i = 0; i < tensor_pols.n(); ++i)
    mapped_point +=
      support_points[renumber[i]] * tensor_pols.compute_value(i, p);

  return mapped_point;
}


template <int dim, int spacedim>
const std::vector<Point<spacedim>> &
MappingQGenericExt<dim, spacedim>::get_support_points() const
{
  return support_points;
}


template <int dim, int spacedim>
std::vector<Point<spacedim>> &
MappingQGenericExt<dim, spacedim>::get_support_points()
{
  return support_points;
}


template <int dim, int spacedim>
std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
MappingQGenericExt<dim, spacedim>::get_data(
  const UpdateFlags      update_flags,
  const Quadrature<dim> &quadrature) const
{
  return MappingQGeneric<dim, spacedim>::get_data(update_flags, quadrature);
}

#endif /* INCLUDE_MAPPING_Q_GENERIC_EXT_H_ */
