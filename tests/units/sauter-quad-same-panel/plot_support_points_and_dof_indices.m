## Support points and DoF indices in Kx:
## DoF_index X Y Z
## 0 0.00000 0.00000 0.00000
## 6 0.500000 0.00000 0.00000
## 1 1.00000 0.00000 0.00000
## 4 0.00000 1.00000 0.00000
## 8 0.500000 1.00000 0.00000
## 5 1.00000 1.00000 0.00000
## 2 0.00000 2.00000 0.00000
## 7 0.500000 2.00000 0.00000
## 3 1.00000 2.00000 0.00000
## Support points and DoF indices in Ky:
## DoF_index X Y Z
## 0 0.00000 0.00000 0.00000
## 6 0.500000 0.00000 0.00000
## 1 1.00000 0.00000 0.00000
## 4 0.00000 1.00000 0.00000
## 8 0.500000 1.00000 0.00000
## 5 1.00000 1.00000 0.00000
## 2 0.00000 2.00000 0.00000
## 7 0.500000 2.00000 0.00000
## 3 1.00000 2.00000 0.00000

clear all;
ConfigGraphicsToolkit;

kx_support_points_and_dof_indices = [0 0.00000 0.00000 0.00000
				     6 0.500000 0.00000 0.00000
				     1 1.00000 0.00000 0.00000
				     4 0.00000 1.00000 0.00000
				     8 0.500000 1.00000 0.00000
				     5 1.00000 1.00000 0.00000
				     2 0.00000 2.00000 0.00000
				     7 0.500000 2.00000 0.00000
				     3 1.00000 2.00000 0.00000];

NewFigure;
Plot3DPointList(kx_support_points_and_dof_indices(:,2:4), "ro-")
hold on;

## Add DoF index labels.
for m = 1:size(kx_support_points_and_dof_indices, 1)
  text(kx_support_points_and_dof_indices(m, 2),
       kx_support_points_and_dof_indices(m, 3),
       kx_support_points_and_dof_indices(m,4),
       num2str(kx_support_points_and_dof_indices(m, 1)), "verticalalignment", "bottom", "horizontalalignment", "right");
endfor

axis on;
axis equal;
view(0, 90);
xlabel("x");
ylabel("y");
title("Support points and DoF indices in Kx");

ky_support_points_and_dof_indices = [0 0.00000 0.00000 0.00000
				     6 0.500000 0.00000 0.00000
				     1 1.00000 0.00000 0.00000
				     4 0.00000 1.00000 0.00000
				     8 0.500000 1.00000 0.00000
				     5 1.00000 1.00000 0.00000
				     2 0.00000 2.00000 0.00000
				     7 0.500000 2.00000 0.00000
				     3 1.00000 2.00000 0.00000];

NewFigure;
Plot3DPointList(ky_support_points_and_dof_indices(:,2:4), "ro-")
hold on;

## Add DoF index labels.
for m = 1:size(ky_support_points_and_dof_indices, 1)
  text(ky_support_points_and_dof_indices(m, 2),
       ky_support_points_and_dof_indices(m, 3),
       ky_support_points_and_dof_indices(m,4),
       num2str(ky_support_points_and_dof_indices(m, 1)), "verticalalignment", "bottom", "horizontalalignment", "right");
endfor

axis on;
axis equal;
view(0, 90);
xlabel("x");
ylabel("y");
title("Support points and DoF indices in Ky");
