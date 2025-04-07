function show_matrix(M, map_name, climits)
  if (!exist("map_name", "var"))
    ## Do not use "jet" anymore, because it has a misleading representation of
    ## data.
    map_name = "viridis";
  endif

  colormap(map_name);

  is_complex = iscomplex(M);
  if (exist("climits", "var"))
    if is_complex
      subplot(1, 2, 1);
      imagesc(real(M), climits);
      colorbar(gca);
      title("Real part");
      axis("on", "tic", "nolabel");
      axis equal;
      
      subplot(1, 2, 2);
      imagesc(imag(M), climits);
      colorbar(gca);
      title("Imaginary part");
      axis("on", "tic", "nolabel");
      axis equal;
      scale_fig(gcf, [2, 1]);
    else
      imagesc(M, climits);

      axis("on", "tic", "nolabel");
      axis equal;
    endif
  else
    if is_complex
      subplot(1, 2, 1);
      imagesc(real(M));
      colorbar(gca);
      title("Real part");
      axis("on", "tic", "nolabel");
      axis equal;
      
      subplot(1, 2, 2);
      imagesc(imag(M));
      colorbar(gca);
      title("Imaginary part");
      axis("on", "tic", "nolabel");
      axis equal;
      scale_fig(gcf, [2, 1]);
    else
      imagesc(M);

      axis("on", "tic", "nolabel");
      axis equal;
    endif
  endif
endfunction
