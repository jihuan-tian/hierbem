function fig_xcord = get_fig_xcord(h)
  if (exist("h", "var"))
    fig_handle = h;
  else
    fig_handle = gcf;
  endif

  fig_position = get(fig_handle, "position");
  fig_xcord =  fig_position(1);
endfunction
