## Declare global variables used for graphics configuration.
DeclareGraphicsGlobals;

## Select default graphical toolkit.
# graphics_toolkit("gnuplot");
graphics_toolkit("qt");

point_size = 1.0 / 72.0;
fig_font_size = 12;
legend_font_size = 8;
fig_line_width = 1;
fig_marker_size = 6;
ax_font_size= 12;
ax_line_width = 1;

set (0, "defaultaxesfontname", "Helvetica");
set (0, "defaultaxesfontsize", ax_font_size);
set (0, "defaulttextfontname", "Helvetica");
set (0, "defaulttextfontsize", fig_font_size);
set(0, "defaultlinelinewidth", fig_line_width);

## Generate a list of colors used for line plotting.
clist = GenColorList();
