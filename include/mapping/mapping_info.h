/**
 * @file mapping_info.h
 * @brief Introduction of mapping_info.h
 *
 * @date 2024-08-09
 * @author Jihuan Tian
 */
#ifndef INCLUDE_MAPPING_MAPPING_INFO_H_
#define INCLUDE_MAPPING_MAPPING_INFO_H_

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_tools.h>

#include "bem_tools.h"
#include "mapping_q_generic_ext.h"

namespace HierBEM
{
  using namespace dealii;

  /**
   * Class holding both the mapping object and its internal data.
   *
   * @tparam dim
   * @tparam spacedim
   */
  template <int dim, int spacedim>
  class MappingInfo
  {
  public:
    /**
     * Default constructor.
     * @pre
     * @post
     */
    MappingInfo();

    /**
     * Constructor.
     *
     * @pre
     * @post
     * @param order
     */
    MappingInfo(const unsigned int order);

    /**
     * Disable the copy constructor, because this class contains a @p unique_ptr
     * member.
     *
     * @pre
     * @post
     * @param
     */
    MappingInfo(const MappingInfo<dim, spacedim> &) = delete;

    /**
     * Disable the copy assignment operator, because this class contains a
     * @p unique_ptr member.
     *
     * @pre
     * @post
     * @param
     * @return
     */
    MappingInfo<dim, spacedim> &
    operator=(const MappingInfo<dim, spacedim> &) = delete;

    /**
     * Resize the data tables in the internal data.
     *
     * The data tables depend on the number of mapping shape functions and the
     * number of quadrature points. After resizing and computing shape function
     * values including their derivatives, the values in the data tables will be
     * copied into BEMValues. Then the internal data can be resized to be used
     * for other cases.
     *
     * @pre
     * @post
     * @param n_q_points
     */
    void
    resize_internal_data(const unsigned int n_q_points) const;

    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData> &
    get_data()
    {
      return data;
    }

    MappingQGenericExt<dim, spacedim> &
    get_mapping()
    {
      return mapping;
    }

    const std::unique_ptr<
      typename MappingQGeneric<dim, spacedim>::InternalData> &
    get_data() const
    {
      return data;
    }

    const MappingQGenericExt<dim, spacedim> &
    get_mapping() const
    {
      return mapping;
    }

    const std::array<std::vector<unsigned int>,
                     GeometryInfo<dim>::vertices_per_cell> &
    get_lexicographic_numberings() const
    {
      return lexicographic_numberings;
    }

    const std::array<std::vector<unsigned int>,
                     GeometryInfo<dim>::vertices_per_cell> &
    get_reversed_lexicographic_numberings() const
    {
      return reversed_lexicographic_numberings;
    }

  private:
    /**
     * Create the internal data object in the parent @p Mapping object.
     *
     * @pre
     * @post
     */
    void
    init_mapping_data();

    /**
     * Initialize the lexicographic numberings and their reversed numberings for
     * accessing mapping support points.
     *
     * @pre
     * @post
     */
    void
    init_lexicographic_numberings_and_reverse();

    /**
     * Mapping object.
     */
    MappingQGenericExt<dim, spacedim> mapping;

    /**
     * Pointer to the InternalData in the mapping.
     */
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData> data;

    /**
     * The numbering used for accessing the list of support points in the
     * mapping object in the lexicographic order by starting from a specific
     * cell vertex, where the list of support points are stored in the
     * hierarchic order.
     *
     * The array index is the vertex index.
     */
    std::array<std::vector<unsigned int>, GeometryInfo<dim>::vertices_per_cell>
      lexicographic_numberings;

    /**
     * The numbering used for accessing the list of support points in the
     * mapping object in the reversed lexicographic order by starting from a
     * specific cell vertex, where the list of support points are stored in the
     * hierarchical order.
     *
     * The array index is the vertex index.
     *
     * \mynote{This numbering occurs only when \f$K_x\f$ and \f$K_y\f$ share a
     * common edge and it is applied to the mapping for \f$K_y\f$.}
     */
    std::array<std::vector<unsigned int>, GeometryInfo<dim>::vertices_per_cell>
      reversed_lexicographic_numberings;
  };


  template <int dim, int spacedim>
  MappingInfo<dim, spacedim>::MappingInfo()
    : mapping(0)
    , data(nullptr)
  {}


  template <int dim, int spacedim>
  MappingInfo<dim, spacedim>::MappingInfo(const unsigned int order)
    : mapping(order)
    , data(nullptr)
  {
    init_mapping_data();
    init_lexicographic_numberings_and_reverse();
  }


  template <int dim, int spacedim>
  void
  MappingInfo<dim, spacedim>::resize_internal_data(
    const unsigned int n_q_points) const
  {
    const unsigned int mapping_n_shape_functions = data->n_shape_functions;

    data->shape_values.resize(mapping_n_shape_functions * n_q_points);
    data->shape_derivatives.resize(mapping_n_shape_functions * n_q_points);
  }


  template <int dim, int spacedim>
  void
  MappingInfo<dim, spacedim>::init_mapping_data()
  {
    std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
      database = mapping.get_data(update_default, QGauss<dim>(1));
    /**
     * Downcast the smart pointer of @p Mapping<dim, spacedim>::InternalDataBase to
     * @p MappingQGeneric<dim,spacedim>::InternalData by first unwrapping
     * the original smart pointer via @p static_cast then wrapping it again.
     */
    data =
      std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
        static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
          database.release()));
  }


  template <int dim, int spacedim>
  void
  MappingInfo<dim, spacedim>::init_lexicographic_numberings_and_reverse()
  {
    // Generate lexicographic numberings.
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      {
        lexicographic_numberings[v].resize(data->n_shape_functions);

        if (v == 0)
          {
            lexicographic_numberings[v] =
              FETools::lexicographic_to_hierarchic_numbering<dim>(
                mapping.get_degree());
          }
        else
          {
            BEMTools::generate_forward_mapping_support_point_permutation(
              mapping, v, lexicographic_numberings[v]);
          }
      }

    // Generate reversed lexicographic numberings.
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      {
        reversed_lexicographic_numberings[v].resize(data->n_shape_functions);

        BEMTools::generate_backward_mapping_support_point_permutation(
          mapping, v, reversed_lexicographic_numberings[v]);
      }
  }
} // namespace HierBEM

#endif /* INCLUDE_MAPPING_MAPPING_INFO_H_ */
