#include "stats.hpp"

int main(int argc, char **argv) {
  Statistics stat = Statistics();
  stat.parse_result(argv[1]);
  stat.summary();
  return 0;
}
