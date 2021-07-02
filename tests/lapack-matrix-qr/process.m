clear all;
load "/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/lapack-matrix-qr/lapack-matrix-qr.output";

norm(M1 - Q1 * R1, "fro")
norm(M2 - Q2 * R2, "fro")
norm(M3 - Q3 * R3, "fro")
