#ifndef INTERVAL_H
#define INTERVAL_H

#include <iostream>

template <class T>
class Interval {
    public:
        friend std::ostream &operator<<(std::ostream &stream, Interval i) {
            stream << "[" << i.start << ", " << i.finish << "] size=" << (i.finish - i.start); return stream;
        }

        Interval(const Interval &src) : start(src.getStart()) , finish(src.getFinish()) {}

        Interval(T start, T finish) {
            this->start = start; this->finish = finish;
        }

        T getStart() const {return this->start;}
        T getFinish() const {return this->finish;}
        bool operator<(const Interval &i) const {
            if (inside(i)) return false;
            if (i.inside(*this)) return false;
            bool res = (this->start <= i.getStart() && (this->finish <= i.getStart()));
            return res;
        }
        bool inside(const Interval &i) const {
            return (this->start >= i.getStart() && (this->finish <= i.getFinish()));
        }
    private:
        T start;
        T finish;
};

#endif
