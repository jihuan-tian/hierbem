function plot_connecting_edge(block1, block2, arrow_length, arrow_width, arrow_type)
  drawArrow((block1(1,1) + block1(3,1)) / 2, (block1(1,2) + block1(3,2)) / 2, (block2(1,1) + block2(3,1)) / 2, (block2(1,2) + block2(3,2)) / 2, arrow_length, arrow_width, arrow_type);
endfunction
