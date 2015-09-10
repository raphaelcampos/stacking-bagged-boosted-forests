#ifndef UTILS_H
#define UTILS_H

#include <cstring>
#include <sstream>

class Utils {
    public:
        template<class T>
        static std::string toString(const T source) {
            std::stringstream ss;
            ss << source;
            return ss.str();
        }

        static void errorMessage(const std::string &msg) {
            std::cerr << "Fatal error: " << msg << std::endl;
            exit(EXIT_FAILURE);
        }

        static void srand(unsigned int seed) {
            ::srand(seed);
        }

        // return a uniform random number in [0..1].
        static double rand() {
            return (double)(::rand() % 1000000) / 1000000.0;
        }

        // return a inform rand number in [a..b].
        static double rand(double a, double b) {
            return (b - a) * Utils::rand() + a;
        }

        // return a uniform random integer value in [0..n-1]
        static unsigned long rand(unsigned long n) {
            if (n < 0) n = -n;
            if (n == 0) return 0;
            unsigned long guard = (unsigned long) (Utils::rand() * n);
            return (guard > n) ? n : guard;
        }
};

#endif
