#include <chrono>
#include <iostream>
#include <stack>
#include <string>
#include <vector>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;
class Stopwatch {
public:
  Stopwatch() = default;
  Stopwatch(Stopwatch &&) = default;
  Stopwatch(const Stopwatch &) = default;
  Stopwatch &operator=(Stopwatch &&) = default;
  Stopwatch &operator=(const Stopwatch &) = default;
  ~Stopwatch();
  inline void start(std::string name) {
    time_point start = std::chrono::high_resolution_clock::now();
    time_points_stack.emplace(name, start);
  }

  inline void end() {
    time_point end = std::chrono::high_resolution_clock::now();
    auto stack_top = time_points_stack.top();
    time_points_stack.pop();
    std::chrono::duration<double, std::milli> ms = end - stack_top.second;
    results_list.emplace_back(stack_top.first, ms.count());
  }

private:
  std::stack<std::pair<std::string, time_point>> time_points_stack{};
  std::vector<std::pair<std::string, double>> results_list{};
};

Stopwatch::~Stopwatch() {
  for (auto result : results_list) {
    std::cerr << result.first << ": " << result.second << "\n";
  }
}
