// Generate a regular grid.
path[] regular_grid(pair origin = (0, 0), pair grid_size, pair cell_num)
{
  path[] grid_path;
  
  real x_cell_size = grid_size.x / cell_num.x;
  real y_cell_size = grid_size.y / cell_num.y;

  // Plot vertical lines.
  for (int i = 0; i <= cell_num.x; ++i)
    {
      real x = i * x_cell_size;
      grid_path = grid_path^^((x, 0) + origin)--((x, grid_size.y) + origin);
    }

  // Plot horizontal lines.
  for (int j = 0; j <= cell_num.y; ++j)
    {
      real y = j * y_cell_size;
      grid_path = grid_path^^((0, y) + origin)--((grid_size.x, y) + origin);
    }

  return grid_path;
}

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

// Generate the path for a matrix block, given its row and column index ranges.
// The direction of the row dimension is the reversed Y axis.
path matrix_block(pair tau, pair sigma)
{
  pair left_bottom = (sigma.x - 0.5, -(tau.y - 0.5));
  pair right_top = left_bottom + (sigma.y - sigma.x, tau.y - tau.x);
  return box(left_bottom, right_top);
}

void plot_matrix_block(picture pic = currentpicture, pair tau, pair sigma, pen fillpen = nullpen, pen drawpen = defaultpen)
{
  path mat = matrix_block(tau, sigma);
  
  if (fillpen != nullpen)
    {
      filldraw(pic, mat, fillpen, drawpen);
    }
  else
    {
      draw(pic, mat, drawpen);
    }
}

void plot_bct_struct(picture pic = currentpicture, string filename, pen near_field_fillpen = rgb(2.6700e-01, 4.8743e-03, 3.2942e-01), pen far_field_fillpen = rgb(0.1306, 0.5577, 0.5522), pen border_pen = rgb(0.9932, 0.9062, 0.1439))
{
  file fin = input(filename);
  int[] tau = {0, 0};
  int[] sigma = {0, 0};
  int is_near_field;
  int rank;

  while(!eof(fin))
    {
      // Remove '[' from the stream.
      string c = getc(fin);

      if (c == '[')
	{
	  tau[0] = fin;
	  tau[1] = fin;

	  // Remove ')', ',' and '[' from the stream.
	  getc(fin);
	  getc(fin);
	  getc(fin);

	  sigma[0] = fin;
	  sigma[1] = fin;

	  // Remove ')' and ',' from the stream.
	  getc(fin);
	  getc(fin);

	  is_near_field = fin;

	  // Remove ',' from the stream.
	  getc(fin);

	  rank = fin;

	  // Remove new line.
	  getc(fin);

	  if (is_near_field == 1)
	    {
	      plot_matrix_block((tau[0], tau[1]), (sigma[0], sigma[1]), near_field_fillpen, border_pen);
	    }
	  else
	    {
	      plot_matrix_block((tau[0], tau[1]), (sigma[0], sigma[1]), far_field_fillpen, border_pen);
	    }
	}
      else
	{
	  break;
	}
    }

  close(fin);
}

void plot_task_edge(picture pic = currentpicture, pair tau1, pair sigma1, pair tau2, pair sigma2, pen p = defaultpen, real arrow_size = 5)
{
  pair center1 = ((sigma1.x + sigma1.y - 1) / 2, (-tau1.x - tau1.y + 1) / 2);
  pair center2 = ((sigma2.x + sigma2.y - 1) / 2, (-tau2.x - tau2.y + 1) / 2);

  dot(center1, p);
  draw(center1--center2, p, Arrow(arrow_size));
}

void plot_diagonal_line(picture pic = currentpicture, pair tau, pair sigma, pen p = defaultpen)
{
  draw((sigma.x - 0.5, 0.5 - tau.x)--(sigma.y - 0.5, -(tau.y - 0.5)), p);
}

void plot_solve_to_update_dependencies(picture pic = currentpicture, string filename, pen fillpen_for_solve_block = orange + opacity(0.5), pen drawpen_for_solve_block = orange, pen fillpen_for_update_block = blue + opacity(0.5), pen drawpen_for_update_block = blue, pen drawpen_for_edge = white, real arrow_size = 5)
{
  file fin = input(filename);

  int[] tau_solve = {0, 0};
  int[] sigma_solve = {0, 0};
  int[] tau_update = {0, 0};
  int[] sigma_update = {0, 0};
  
  while(!eof(fin))
    {
      string c = getc(fin);
      
      if (c == '[')
	{
	  tau_solve[0] = fin;
	  getc(fin);
	  tau_solve[1] = fin;
	  getc(fin);
	  getc(fin);
	  getc(fin);
	  getc(fin);
	  sigma_solve[0] = fin;
	  getc(fin);
	  sigma_solve[1] = fin;
	  getc(fin);

	  plot_matrix_block(pic, (tau_solve[0], tau_solve[1]), (sigma_solve[0], sigma_solve[1]), fillpen_for_solve_block, drawpen_for_solve_block);
	  
	  getc(fin);
	  getc(fin);
	  getc(fin);
	  getc(fin);
	  getc(fin);

	  getc(fin);
	  tau_update[0] = fin;
	  getc(fin);
	  tau_update[1] = fin;
	  getc(fin);
	  getc(fin);
	  getc(fin);
	  getc(fin);
	  sigma_update[0] = fin;
	  getc(fin);
	  sigma_update[1] = fin;
	  getc(fin);

	  plot_matrix_block(pic, (tau_update[0], tau_update[1]), (sigma_update[0], sigma_update[1]), fillpen_for_update_block, drawpen_for_update_block);

	  plot_task_edge(pic, (tau_solve[0], tau_solve[1]), (sigma_solve[0], sigma_solve[1]), (tau_update[0], tau_update[1]), (sigma_update[0], sigma_update[1]), drawpen_for_edge, arrow_size);
	}
    }
}
