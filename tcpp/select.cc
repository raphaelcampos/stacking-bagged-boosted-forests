#include "selector.hpp"

int main(int argc, char **argv) {
  KNN_Selector s = KNN_Selector(argv[1], argv[2]);
  s.select(atoi(argv[3]), false);
  return 0;
}
