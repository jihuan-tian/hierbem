#!/bin/bash

/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -n 4 -e 0.1 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/01.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -n 4 -e 0.01 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/02.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -n 8 -e 0.1 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/03.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -n 8 -e 0.01 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/04.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -n 16 -e 0.1 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/05.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -n 16 -e 0.01 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/06.dat
