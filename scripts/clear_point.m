function clear_point(h, evt, current_axes)
  list_of_markder_handles = get(current_axes, "userdata");

  if (length(list_of_markder_handles) > 0)
    delete(list_of_markder_handles);
    set(current_axes, "userdata", []);
  endif
endfunction
