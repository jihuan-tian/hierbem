cl__1 = 1;
Point(1) = {0, 0, 0, cl__1};
Point(2) = {2, 0, 0, cl__1};
Point(3) = {2, 2, 0, cl__1};
Point(4) = {1.5, 2, 0, cl__1};
Point(5) = {1.5, 0.5, 0, cl__1};
Point(6) = {0, 0.5, 0, cl__1};
Point(7) = {0, 0, 1, cl__1};
Point(8) = {2, 0, 1, cl__1};
Point(9) = {2, 2, 1, cl__1};
Point(10) = {1.5, 2, 1, cl__1};
Point(11) = {1.5, 0.5, 1, cl__1};
Point(12) = {0, 0.5, 1, cl__1};
Line(1) = {6, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
Line(5) = {4, 5};
Line(6) = {5, 6};
Line(7) = {12, 6};
Line(8) = {12, 7};
Line(9) = {7, 1};
Line(10) = {7, 8};
Line(11) = {8, 2};
Line(12) = {8, 9};
Line(13) = {11, 10};
Line(14) = {11, 12};
Line(15) = {10, 9};
Line(16) = {11, 5};
Line(17) = {9, 3};
Line(18) = {10, 4};
Curve Loop(1) = {6, 1, 2, 3, 4, 5};
Plane Surface(1) = {1};
Curve Loop(2) = {14, 8, 10, 12, -15, -13};
Plane Surface(2) = {2};
Curve Loop(3) = {7, 1, -9, -8};
Plane Surface(3) = {3};
Curve Loop(4) = {9, 2, -11, -10};
Plane Surface(4) = {4};
Curve Loop(5) = {12, 17, -3, -11};
Plane Surface(5) = {5};
Curve Loop(6) = {15, 17, 4, -18};
Plane Surface(6) = {6};
Curve Loop(7) = {13, 18, 5, -16};
Plane Surface(7) = {7};
Curve Loop(8) = {14, 7, -6, -16};
Plane Surface(8) = {8};
Surface Loop(1) = {3, 8, 2, 4, 1, 5, 6, 7};
Volume(1) = {1};
//+
Physical Volume(1) = {1};
//+
Physical Surface(1) = {3};
//+
Physical Surface(2) = {6};
//+
Physical Surface(19) = {2};
//+
Physical Surface(20) = {1};
//+
Physical Surface(21) = {8};
//+
Physical Surface(22) = {4};
//+
Physical Surface(23) = {7};
//+
Physical Surface(24) = {5};
