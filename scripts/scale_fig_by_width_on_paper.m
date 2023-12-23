function scale_fig_by_width_on_paper(h, new_width_cm)
  new_fig_size_inch = calc_fig_size_by_width_on_paper(h, new_width_cm);
  set_fig_size_on_paper(h, new_fig_size_inch);
endfunction
