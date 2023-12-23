function scale_fig(h, scale_factor)
  ## scale_fig - This function should be called after the plotting is finished.
  
  switch (length(scale_factor))
    case 1
      ## Scale the figure with a same factor for width and height.
      set_fig_size(h, get_fig_size(h) * scale_factor);
    case 2
      ## Scale the figure with different factors for width and height.
      set_fig_size(h, get_fig_size(h) .* scale_factor);
  endswitch
endfunction
