/**
 * @file aca_plus.cpp
 * @brief Introduction of aca_plus.cpp
 *
 * @date 2022-03-09
 * @author Jihuan Tian
 */

#include "hmatrix/aca_plus/aca_config.h"

#include "config.h"

HBEM_NS_OPEN

ACAConfig::ACAConfig()
  : max_iter(0)
  , epsilon(0.)
  , eta(0.)
{}


ACAConfig::ACAConfig(unsigned int v_max_iter, double v_epsilon, double v_eta)
  : max_iter(v_max_iter)
  , epsilon(v_epsilon)
  , eta(v_eta)
{}

HBEM_NS_CLOSE
