function set_fig_size(h, fig_size)
  set(h, "position", [get_fig_xcord(h), get_fig_ycord(h), fig_size(1), fig_size(2)]);
  set(h, "paperpositionmode", "auto");
endfunction
