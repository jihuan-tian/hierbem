# Producer: sauter_quadrature_for_near_field_matrix_entries_producer

Workflow of the producer thread function in `include/quadrature/sauter_quadrature.hcu` (both overloads share this control flow).

```mermaid
flowchart TD
  Start([Start])
  Init["Create per-thread scratch data (PairCellWiseScratchData, PairCellWisePerTaskData; mass-matrix overload also creates CellWiseScratchDataForMassMatrix)"]
  ForLeaf["For each leaf index l in [start_leaf_set_index, end_leaf_set_index)"]
  GetLeaf["Get leaf H-matrix, full matrix, row/col index ranges"]
  SymCheck{"enable_build_symmetric_hmat and is_kernel_symmetric?"}
  BlockSwitch["Switch on leaf_mat->get_block_type()"]
  DiagBlock["Diagonal block: loop (row, col) with col <= row"]
  UpperBlock["Upper triangular block: skip"]
  LowerBlock["Lower triangular block: loop all (row, col)"]
  UndefBlock["Undefined block: Assert and break"]
  FullMatrix["Else: loop all (row, col) in full matrix"]
  CreateTask["For each (row,col): create_sauter_quadrature_tasks_on_one_pair_of_dofs"]
  AddToBuffer["Add task to one of: same_panel, common_edge, common_vertex, regular ring buffer"]
  NextLeaf["Next leaf l"]
  Release["Release scratch_data and copy_data"]
  Exit([Producer exits])

  Start --> Init
  Init --> ForLeaf
  ForLeaf --> GetLeaf
  GetLeaf --> SymCheck
  SymCheck -->|Yes| BlockSwitch
  SymCheck -->|No| FullMatrix
  BlockSwitch --> DiagBlock
  BlockSwitch --> UpperBlock
  BlockSwitch --> LowerBlock
  BlockSwitch --> UndefBlock
  DiagBlock --> CreateTask
  LowerBlock --> CreateTask
  FullMatrix --> CreateTask
  CreateTask --> AddToBuffer
  AddToBuffer --> NextLeaf
  UpperBlock --> NextLeaf
  UndefBlock --> NextLeaf
  NextLeaf --> ForLeaf
  ForLeaf -->|Done| Release
  Release --> Exit
```
