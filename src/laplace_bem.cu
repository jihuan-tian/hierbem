#include "config.h"
#include "laplace_bem.impl.hcu"
#include "utility/template_instantiator.h"

HBEM_NS_OPEN

// TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_2((LaplaceBEM),
//                                         TI_INST_BOUNDARY_DIMS,
//                                         TI_INST_SPACE_DIMS)
template class LaplaceBEM<2, 3>;

HBEM_NS_CLOSE
