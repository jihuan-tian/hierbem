function fig_ycord = get_fig_ycord(h)
  if (exist("h", "var"))
    fig_handle = h;
  else
    fig_handle = gcf;
  endif

  fig_position = get(fig_handle, "position");
  fig_ycord =  fig_position(2);
endfunction
