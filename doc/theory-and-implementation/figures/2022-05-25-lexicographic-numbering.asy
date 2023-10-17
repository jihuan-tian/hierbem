unitsize(0.5cm);
real grid_spacing = 4;
int grid_num = 4;

int[] dof_indices;
for(int i=0;i<grid_num*grid_num;++i) {
  dof_indices.push(i);
}

// Plot the grid in deal.ii's lexicographic order.
int counter = 0;
for(int j=0;j<grid_num;++j){
  for(int i=0;i<grid_num;++i) {
    pair circle_center = (i,j) * grid_spacing;
    draw(shift(circle_center)*unitcircle);
    // Add label at the circle center.
    label(format("%d", counter), circle_center);
    ++counter;
  }
}
