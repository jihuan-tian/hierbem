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

    const std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
      &
      get_data() const
    {
      return data;
    }

    const MappingQGenericExt<dim, spacedim> &
    get_mapping() const
    {
      return mapping;
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

    MappingQGenericExt<dim, spacedim> mapping;
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData> data;
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
} // namespace HierBEM

#endif /* INCLUDE_MAPPING_MAPPING_INFO_H_ */
