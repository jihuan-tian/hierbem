function scale_fig_by_height_on_paper(h, new_height_cm)
  new_fig_size_inch = calc_fig_size_by_height_on_paper(h, new_height_cm);
  set_fig_size_on_paper(h, new_fig_size_inch);
endfunction
