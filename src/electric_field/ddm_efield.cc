/**
 * @file ddm_efield.cc
 * @brief Introduction of ddm_efield.cc
 *
 * @date 2024-08-03
 * @author Jihuan Tian
 */

#include "electric_field/ddm_efield.h"

namespace HierBEM
{
  void
  SubdomainTopology::generate_topology(
    const std::string &cad_file,
    const std::string &mesh_file,
    const double       eps_for_orientation_detection)
  {
    gmsh::initialize();
    gmsh::option::setNumber("General.Verbosity", 0);
    gmsh::open(cad_file);
    gmsh::merge(mesh_file);
    gmsh::model::occ::synchronize();

    // At the moment, we only support 3D model.
    AssertDimension(gmsh::model::getDimension(), 3);

    // Get all 3D volume entities.
    gmsh::vectorpair volume_dimtag_list;
    gmsh::model::occ::getEntities(volume_dimtag_list, 3);

    // The boundary entities of each 3D volume entity.
    std::vector<EntityTag> oriented_surface_tags;
    for (const auto &volume_dimtag : volume_dimtag_list)
      {
        GmshManip::get_oriented_volume_boundaries(
          volume_dimtag.second,
          oriented_surface_tags,
          face_to_subdomain,
          eps_for_orientation_detection);

        subdomain_to_face[volume_dimtag.second] = oriented_surface_tags;
      }

    gmsh::clear();
    gmsh::finalize();
  }


  void
  SubdomainTopology::print(std::ostream &out) const
  {
    out << "=== subdomain-to-face ===\n";

    for (const auto &record : subdomain_to_face)
      {
        out << record.first << ":";
        for (const auto face_tag : record.second)
          {
            out << " " << face_tag;
          }
        out << "\n";
      }

    out << "=== face-to-subdomain ===\n";
    for (const auto &record : face_to_subdomain)
      {
        out << record.first << ": " << record.second[0] << " "
            << record.second[1] << "\n";
      }

    out << std::endl;
  }
} // namespace HierBEM
