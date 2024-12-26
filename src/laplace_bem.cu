#include "config.h"
#include "laplace_bem.impl.hcu"
#include "build/template_instantiator.h"

HBEM_NS_OPEN

TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_2((LaplaceBEM),
                                        TI_INST_BOUNDARY_DIMS,
                                        TI_INST_SPACE_DIMS)

HBEM_NS_CLOSE
