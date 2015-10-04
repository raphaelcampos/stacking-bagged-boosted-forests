#ifndef OUTPUTER_HPP__
#define OUTPUTER_HPP__

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "scores.hpp"
#include "utils.hpp"

class Outputer {
 public:
  Outputer() {}

  virtual void output(const std::string &ln) {
    #pragma omp critical(output)
    {
      printf("%s\n", ln.data());
    }   
  }

  virtual void output(Scores<double> &sco, const double normalizer=1) {
    std::stringstream ss;
    ss << sco.get_id() << " " << sco.get_class();
    while(!sco.empty()) {
      Similarity<double> s = sco.top();
      ss << " " << s.class_name << ":" << (s.similarity/normalizer);
      sco.pop();
      break;
    }

    #pragma omp critical(output)
    {
      printf("%s\n", ss.str().data());
    }
  }

  virtual void output(Scores<MyBig> &sco, const MyBig &normalizer="1.0") {
    std::stringstream ss;
    ss << sco.get_id() << " " << sco.get_class();
    while(!sco.empty()) {
      Similarity<MyBig> s = sco.top();
      ss << " " << s.class_name << ":" << (s.similarity/normalizer);
      sco.pop();
    }

    #pragma omp critical(output)
    {
      printf("%s\n", ss.str().data());
    }
  }

  virtual ~Outputer() {}
};

class FileOutputer : public Outputer {
 public:
  FileOutputer(const std::string &fn) : os_(fn.data()) {
    #pragma omp critical (output)
    {
      #pragma omp flush(os_)
      if (!os_.is_open()) os_.open(fn.data());
    }
  }

  virtual void output(const std::string &ln) {
    std::string res = ln + std::string("\n");
    #pragma omp critical(output)
    {
      os_ << res << std::flush;
    }   
  }

  virtual void output(Scores<double> &sco, const double normalizer=1.0) {
    std::stringstream ss;
    ss << sco.get_id() << " " << sco.get_class();
    while(!sco.empty()) {
      Similarity<double> s = sco.top();
      ss << " " << s.class_name << ":" << (s.similarity/normalizer);
      sco.pop();
    }
    ss << std::endl;

    #pragma omp critical (output)
    {
      #pragma omp flush(os_)
      os_ << ss.str() << std::flush;
    }
  }

  virtual void output(Scores<MyBig> &sco, const MyBig &normalizer="1.0") {
    std::stringstream ss;
    ss << sco.get_id() << " " << sco.get_class();
    while(!sco.empty()) {
      Similarity<MyBig> s = sco.top();
      ss << " " << s.class_name << ":" << (s.similarity/normalizer);
      sco.pop();
    }
    ss << std::endl;

    #pragma omp critical (output)
    {
      #pragma omp flush(os_)
      os_ << ss.str() << std::flush;
    }
  }

  virtual ~FileOutputer() {
    #pragma omp critical (output)
    {
      #pragma omp flush(os_)
      os_.close();
    }
  }

 protected:
  std::ofstream os_;
};

class BufferedOutputer : public Outputer {
 public:
  BufferedOutputer(size_t bf_sz) : bf_sz_(bf_sz) {
    buffer_.reserve(bf_sz);
  }

  virtual void output(Scores<double> &sco) {
    if (buffer_.size() == bf_sz_) {
      #pragma omp critical (output)
      {
        for (int i = 0; i < static_cast<int>(buffer_.size()); i++) {
          printf("%s\n", buffer_[i].data());
        }
        buffer_.clear();
      }
    }
    std::stringstream ss;
    ss << sco.get_id() << " " << sco.get_class();
    while(!sco.empty()) {
      Similarity<double> s = sco.top();
      ss << " " << s.class_name << ":" << s.similarity;
      sco.pop();
    }
    #pragma omp critical (output)
    {
      buffer_.push_back(ss.str());
    }
  }

  virtual void output(Scores<MyBig> &sco) {
     if (buffer_.size() == bf_sz_) {
      #pragma omp critical (output)
      {
        for (int i = 0; i < static_cast<int>(buffer_.size()); i++) {
          printf("%s\n", buffer_[i].data());
        }
        buffer_.clear();
      }
    }
    std::stringstream ss;
    ss << sco.get_id() << " " << sco.get_class();
    while(!sco.empty()) {
      Similarity<MyBig> s = sco.top();
      ss << " " << s.class_name << ":" << s.similarity;
      sco.pop();
    }
    #pragma omp critical (output)
    {
      buffer_.push_back(ss.str()); 
    }
  }

  ~BufferedOutputer() {
    #pragma omp critical (output)
    {
      for (int i = 0; i < static_cast<int>(buffer_.size()); i++) {
        printf("%s\n", buffer_[i].data());
      }
      buffer_.clear();
    }
  }

 protected:
  size_t bf_sz_;
  std::vector<std::string> buffer_;
};

#endif
