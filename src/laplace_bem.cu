#include "build_tools/template_instantiator.h"
#include "config.h"
#include "laplace_bem.impl.hcu"

HBEM_NS_OPEN

TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_4_WITH_RANGE_KERNEL_NUMBER_TYPES_(
  (LaplaceBEM),
  TI_INST_BOUNDARY_DIMS,
  TI_INST_SPACE_DIMS,
  TI_GENERATE_ALL_LAPLACE_TYPES_(TI_INST_NUM_TYPES));

HBEM_NS_CLOSE
