#include "ttest.hpp"

int main(int argc, char **argv) {
  TTest test = TTest(std::string(argv[1]), std::string(argv[2]));
  ConfInt ci_mic = test.ttest_mic_f1();
  ConfInt ci_mac = test.ttest_mac_f1();

  std::cout << "MicroF1: [" << ci_mic.lower_bound() << " , " << ci_mic.upper_bound() << "]" << std::endl;
  std::cout << "MacroF1: [" << ci_mac.lower_bound() << " , " << ci_mac.upper_bound() << "]" << std::endl;
  return 0;
}
