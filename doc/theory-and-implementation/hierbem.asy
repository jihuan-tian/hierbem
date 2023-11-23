// Generate a \f$R^p\f$ matrix block in an H-matrix with the standard partition.
path[] rp_block(pair left_bottom_corner, real unit_cell_size, int current_level, int max_level)
{
  real block_size = unit_cell_size * 2^(max_level - current_level);

  return box(left_bottom_corner, left_bottom_corner + (block_size, block_size));
}

// Generate the \f$N^p\f$ matrix block in an H-matrix with the standard partition.
path[] np_block(pair left_bottom_corner, real unit_cell_size, int current_level, int max_level)
{
  path[] hmat_hierarchy;
  real block_size = unit_cell_size * 2^(max_level - current_level);

  hmat_hierarchy = hmat_hierarchy^^box(left_bottom_corner, left_bottom_corner + (block_size, block_size));

  if (current_level != max_level)
    {
      hmat_hierarchy = hmat_hierarchy^^rp_block((left_bottom_corner.x, left_bottom_corner.y + block_size / 2), unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^rp_block((left_bottom_corner.x + block_size / 2, left_bottom_corner.y + block_size / 2), unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^np_block(left_bottom_corner, unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^rp_block((left_bottom_corner.x + block_size / 2, left_bottom_corner.y), unit_cell_size, current_level + 1, max_level);
    }

  return hmat_hierarchy;
}

// Generate the \f$N^{p*}\f$ matrix block in an H-matrix with the standard partition.
path[] np_star_block(pair left_bottom_corner, real unit_cell_size, int current_level, int max_level)
{
  path[] hmat_hierarchy;
  real block_size = unit_cell_size * 2^(max_level - current_level);

  hmat_hierarchy = hmat_hierarchy^^box(left_bottom_corner, left_bottom_corner + (block_size, block_size));

  if (current_level != max_level)
    {
      hmat_hierarchy = hmat_hierarchy^^rp_block((left_bottom_corner.x, left_bottom_corner.y + block_size / 2), unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^np_star_block((left_bottom_corner.x + block_size / 2, left_bottom_corner.y + block_size / 2), unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^rp_block(left_bottom_corner, unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^rp_block((left_bottom_corner.x + block_size / 2, left_bottom_corner.y), unit_cell_size, current_level + 1, max_level);
    }

  return hmat_hierarchy;
}

// Generate the hierarchical structure of an H-matrix using the coarse standard
// partition.
path[] hmat_coarse_std_partition(pair left_bottom_corner, real unit_cell_size, int current_level, int max_level)
{
  path [] hmat_hierarchy;
  real block_size = unit_cell_size * 2^(max_level - current_level);

  hmat_hierarchy = hmat_hierarchy^^box(left_bottom_corner, left_bottom_corner + (block_size, block_size));

  if (current_level != max_level)
    {
      hmat_hierarchy = hmat_hierarchy^^hmat_coarse_std_partition((left_bottom_corner.x, left_bottom_corner.y + block_size / 2.0), unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^rp_block((left_bottom_corner.x + block_size / 2.0, left_bottom_corner.y + block_size / 2.0), unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^rp_block((left_bottom_corner.x, left_bottom_corner.y), unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^hmat_coarse_std_partition((left_bottom_corner.x + block_size / 2.0, left_bottom_corner.y), unit_cell_size, current_level + 1, max_level);
    }

  return hmat_hierarchy;
}

// Generate the hierarchical structure of an H-matrix using the coarse standard
// partition. The total size of the top level matrix block is fixed.
path[] hmat_coarse_std_partition_fixed_size(pair left_bottom_corner, real total_size, int max_level)
{
  real unit_cell_size = total_size / 2^max_level;
  
  return hmat_coarse_std_partition(left_bottom_corner, unit_cell_size, 0, max_level);
}

// Generate the hierarchical structure of an H-matrix using the fine standard
// partition.
path[] hmat_fine_std_partition(pair left_bottom_corner, real unit_cell_size, int current_level, int max_level)
{
  path [] hmat_hierarchy;
  real block_size = unit_cell_size * 2^(max_level - current_level);

  hmat_hierarchy = hmat_hierarchy^^box(left_bottom_corner, left_bottom_corner + (block_size, block_size));

  if (current_level != max_level)
    {
      hmat_hierarchy = hmat_hierarchy^^hmat_fine_std_partition((left_bottom_corner.x, left_bottom_corner.y + block_size / 2.0), unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^np_block((left_bottom_corner.x + block_size / 2.0, left_bottom_corner.y + block_size / 2.0), unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^np_star_block((left_bottom_corner.x, left_bottom_corner.y), unit_cell_size, current_level + 1, max_level);
      hmat_hierarchy = hmat_hierarchy^^hmat_fine_std_partition((left_bottom_corner.x + block_size / 2.0, left_bottom_corner.y), unit_cell_size, current_level + 1, max_level);
    }

  return hmat_hierarchy;
}

// Generate the hierarchical structure of an H-matrix using the fine standard
// partition. The total size of the top level matrix block is fixed.
path[] hmat_fine_std_partition_fixed_size(pair left_bottom_corner, real total_size, int max_level)
{
  real unit_cell_size = total_size / 2^max_level;
  
  return hmat_fine_std_partition(left_bottom_corner, unit_cell_size, 0, max_level);
}
