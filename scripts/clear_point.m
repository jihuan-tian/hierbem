## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function clear_point(h, evt, current_axes)
  list_of_markder_handles = get(current_axes, "userdata");

  if (length(list_of_markder_handles) > 0)
    delete(list_of_markder_handles);
    set(current_axes, "userdata", []);
  endif
endfunction
