# create_sauter_quadrature_tasks_on_one_pair_of_dofs

Workflow for creating Sauter quadrature tasks for one pair of global DoFs. Defined in `include/quadrature/sauter_quadrature.hcu`. Two overloads: without mass matrix (~line 6553) and with mass matrix (~line 6898); control flow is the same except for the optional mass-matrix entry computation.

```mermaid
flowchart TD
  Start([Start: one pair i_global_dof_index, j_global_dof_index])
  ForKx["For each cell kx_cell in support of DoF i (kx_dof_to_cell_topo[i])"]
  KxMapping["Get kx mapping index, mapping info, n_shape_functions, is_kx_normal_inward"]
  KxSupport["Copy of kx mapping: compute_mapping_support_points(kx_cell); copy into scratch_data.kx_mapping_support_points_in_default_order"]
  KxDoFs["kx_cell->get_dof_indices(scratch_data.kx_local_dof_indices_in_default_dof_order)"]
  ForKy["For each cell ky_cell in support of DoF j (ky_dof_to_cell_topo[j])"]
  DetectType["detect_cell_neighboring_type_for_same_triangulations(kx_cell, ky_cell) -> cell_neighboring_type"]
  KyMapping["Get ky mapping index, mapping info, n_shape_functions, is_ky_normal_inward"]
  KySupport["Copy of ky mapping: compute_mapping_support_points(ky_cell); copy into scratch_data.ky_mapping_support_points_in_default_order"]
  KyDoFs["ky_cell->get_dof_indices(scratch_data.ky_local_dof_indices_in_default_dof_order)"]
  Permute["permute_dofs_and_mapping_support_points_for_sauter_quad(scratch_data, copy_data, cell_neighboring_type, kx_cell, ky_cell, kx_mapping_info, ky_mapping_info)"]
  MassCheck{"Mass-matrix overload: kx_cell_index == ky_cell_index and mass_matrix_factor != 0?"}
  CalcMass["Compute mass_matrix_entry: reinit FE values for test/trial space, loop quadrature points, accumulate shape_value * JxW; scale by mass_matrix_factor"]
  MassZero["mass_matrix_entry = 0 (or skip in no-mass overload)"]
  FindIndices["Find i_index in copy_data.kx_local_dof_indices_permuted, j_index in copy_data.ky_local_dof_indices_permuted"]
  SwitchType["Switch on cell_neighboring_type"]
  SamePanel["ring_buffer_for_same_panel.add_task(..., mass_matrix_entry, scratch_data)"]
  CommonEdge["ring_buffer_for_common_edge.add_task(..., mass_matrix_entry, scratch_data)"]
  CommonVertex["ring_buffer_for_common_vertex.add_task(..., mass_matrix_entry, scratch_data)"]
  Regular["ring_buffer_for_regular.add_task(..., mass_matrix_entry, scratch_data)"]
  NextKy["Next ky_cell"]
  NextKx["Next kx_cell"]
  Exit([Return])

  Start --> ForKx
  ForKx --> KxMapping
  KxMapping --> KxSupport
  KxSupport --> KxDoFs
  KxDoFs --> ForKy
  ForKy --> DetectType
  DetectType --> KyMapping
  KyMapping --> KySupport
  KySupport --> KyDoFs
  KyDoFs --> Permute
  Permute --> MassCheck
  MassCheck -->|Yes| CalcMass
  MassCheck -->|No| MassZero
  CalcMass --> FindIndices
  MassZero --> FindIndices
  FindIndices --> SwitchType
  SwitchType --> SamePanel
  SwitchType --> CommonEdge
  SwitchType --> CommonVertex
  SwitchType --> Regular
  SamePanel --> NextKy
  CommonEdge --> NextKy
  CommonVertex --> NextKy
  Regular --> NextKy
  NextKy --> ForKy
  ForKy -->|Done| NextKx
  NextKx --> ForKx
  ForKx -->|Done| Exit
```

## Notes

- **Same panel / common edge / common vertex / regular:** The cell neighboring type selects which of the four ring buffers receives the task. Each task carries scalar data (i_index, j_index, mapping indices, shape function counts, normal orientation) and a mass matrix entry (0 in the no-mass overload).
- **Mass-matrix overload only:** When both cells are the same and `mass_matrix_factor != 0`, the function computes the FEM mass matrix entry for the (i,j) DoF pair on that cell and passes it into `add_task`; the consumer will add this to the Sauter quadrature result.
- **Permutation:** `permute_dofs_and_mapping_support_points_for_sauter_quad` orders DoFs and support points according to the Sauter quadrature rules for the given cell neighboring type; `i_index` and `j_index` refer to these permuted orderings.
