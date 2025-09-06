/**
 * @file mapping_q_ext.h
 * @brief Extend the class @p MappingQ
 *
 * @date 2022-07-08
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_MAPPING_MAPPING_Q_EXT_H_
#define HIERBEM_INCLUDE_MAPPING_MAPPING_Q_EXT_H_

#include <deal.II/base/point.h>
#include <deal.II/base/tensor_product_polynomials.h>

#include <deal.II/fe/fe_data.h>
#include <deal.II/fe/fe_q_base.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim = dim>
class MappingQExt : public MappingQ<dim, spacedim>
{
public:
  /**
   * Default constructor
   */
  MappingQExt();

  MappingQExt(const unsigned int polynomial_degree);

  /**
   * Copy constructor
   *
   * @param mapping
   */
  MappingQExt(const MappingQExt<dim, spacedim> &mapping);

  /**
   * Compute a list of support points in the real cell in the hierarchic
   * order.
   *
   * \alert{Because this function has non-covariant return type and different
   * cv-qualifier (non-const here) from the function with the same name in the
   * parent class, it overloads but not overrides the function in the parent
   * class. So here we explicitly declare the function in the parent class via
   * @p using, which eliminates the compiler warning @p -Woverloaded-virtual.}
   *
   * @param cell
   */
  using MappingQ<dim, spacedim>::compute_mapping_support_points;
  void
  compute_mapping_support_points(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell);

  /**
   * Transform the point in the unit to the real cell, assuming that the
   * @p support_points have already been computed using the function
   * @p MappingQExt<dim, spacedim>::compute_mapping_support_points.
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
  using MappingQ<dim, spacedim>::transform_unit_to_real_cell;
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
   * List of support points in the hierarchic ordering of the current real
   * cell.
   */
  std::vector<Point<spacedim>> support_points;
};


template <int dim, int spacedim>
MappingQExt<dim, spacedim>::MappingQExt()
  : MappingQ<dim, spacedim>(1)
  , support_points(0)
{}


template <int dim, int spacedim>
MappingQExt<dim, spacedim>::MappingQExt(const unsigned int polynomial_degree)
  : MappingQ<dim, spacedim>(polynomial_degree)
  , support_points(0)
{}


template <int dim, int spacedim>
MappingQExt<dim, spacedim>::MappingQExt(
  const MappingQExt<dim, spacedim> &mapping)
  : MappingQ<dim, spacedim>(mapping)
  , support_points(mapping.support_points)
{}


template <int dim, int spacedim>
void
MappingQExt<dim, spacedim>::compute_mapping_support_points(
  const typename Triangulation<dim, spacedim>::cell_iterator &cell)
{
  support_points =
    MappingQ<dim, spacedim>::compute_mapping_support_points(cell);
}


template <int dim, int spacedim>
Point<spacedim>
MappingQExt<dim, spacedim>::transform_unit_to_real_cell(
  const Point<dim> &p) const
{
  // Set up the polynomial space in the lexicographic order.
  const TensorProductPolynomials<dim> tensor_pols(
    Polynomials::generate_complete_Lagrange_basis(this->line_support_points));
  Assert(tensor_pols.n() ==
           Utilities::fixed_power<dim>(this->polynomial_degree + 1),
         ExcInternalError());

  // Get the numbering for accessing the support points in the lexicographic
  // ordering which are stored in the hierarchic ordering.
  const std::vector<unsigned int> renumber(
    FETools::lexicographic_to_hierarchic_numbering<dim>(
      this->polynomial_degree));

  Point<spacedim> mapped_point;
  for (unsigned int i = 0; i < tensor_pols.n(); ++i)
    mapped_point +=
      support_points[renumber[i]] * tensor_pols.compute_value(i, p);

  return mapped_point;
}


template <int dim, int spacedim>
const std::vector<Point<spacedim>> &
MappingQExt<dim, spacedim>::get_support_points() const
{
  return support_points;
}


template <int dim, int spacedim>
std::vector<Point<spacedim>> &
MappingQExt<dim, spacedim>::get_support_points()
{
  return support_points;
}


template <int dim, int spacedim>
std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
MappingQExt<dim, spacedim>::get_data(const UpdateFlags      update_flags,
                                     const Quadrature<dim> &quadrature) const
{
  return MappingQ<dim, spacedim>::get_data(update_flags, quadrature);
}

DEAL_II_NAMESPACE_CLOSE

#endif // HIERBEM_INCLUDE_MAPPING_MAPPING_Q_EXT_H_
