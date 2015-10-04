#ifndef ITI_INCREMENTAL_H__
#define ITI_INCREMENTAL_H__

#include "iti.hpp"

class iti_incremental : public iti{
  public:
  iti_incremental(unsigned int r) : iti(r) {}
  iti_incremental() {}
  void add_instance(instance_object* inst) {tree->add_instance(inst);}
};

#endif
