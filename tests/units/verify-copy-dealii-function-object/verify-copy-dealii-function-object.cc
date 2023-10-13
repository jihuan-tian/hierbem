/**
 * \file verify-copy-dealii-function-object.cc
 * \brief Verify the evaluation of a inherited deal.ii function after being
 * copied to the parent class @p Function.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-06-22
 */

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <iostream>

using namespace dealii;

class DirichletBC : public Function<3>
{
public:
  DirichletBC()
    : Function<3>()
    , x0(0.25, 0.25, 0.25)
  {}

  DirichletBC(const Point<3> &x0)
    : Function<3>()
    , x0(x0)
  {}

  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;
    return 1.0 / 4.0 / numbers::PI / (p - x0).norm();
  }

private:
  Point<3> x0;
};

int
main()
{
  DirichletBC dirichlet_bc;

  /**
   * Evaluate the inherited function object.
   */
  std::cout << "DirichletBC(0,0,0)=" << dirichlet_bc.value(Point<3>(0, 0, 0))
            << std::endl;

  /**
   * @p Function is an abstract class, which cannot be directly created.
   */
  // Function<3> func(dirichlet_bc);

  /**
   * Then we create a reference type of @p Function instead. Because the member
   * function @p value is virtual, polymorphism is enabled. Hence, the function
   * can be correctly evaluated.
   */
  Function<3> &func_ref = dirichlet_bc;
  std::cout << "Func_ref(0,0,0)=" << func_ref.value(Point<3>(0, 0, 0))
            << std::endl;

  /**
   * We further create a pointer type of @p Function, which should also work.
   */
  Function<3> *func_ptr = &dirichlet_bc;
  std::cout << "Func_ptr(0,0,0)=" << func_ptr->value(Point<3>(0, 0, 0))
            << std::endl;

  return 0;
}
