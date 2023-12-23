function scale_fig_on_paper(h, scale_factor)
  switch (length(scale_factor))
    case 1
      set_fig_size_on_paper(h, get_fig_size_on_paper(h) * scale_factor);
    case 2
      set_fig_size_on_paper(h, get_fig_size_on_paper(h) .* scale_factor);
  endswitch
endfunction
