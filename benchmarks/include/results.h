//
// Created by egi on 4/25/20.
//

#ifndef CULIB_RESULTS_H
#define CULIB_RESULTS_H

#include <unordered_map>
#include <vector>

using size_clk   = std::pair<unsigned int, unsigned long long int>; // n + clk
using clk_result = std::unordered_map<std::string, std::vector<size_clk>>;   // type -> size clk
using imp_result = std::unordered_map<std::string, clk_result>; // implementations -> data types
using alg_result = std::unordered_map<std::string, imp_result>; // algorithm -> implementations
using scp_result = std::unordered_map<std::string, alg_result>; // scope -> algorithms
using dev_result = std::unordered_map<std::string, scp_result>; // device -> scopes

void dump_results (const dev_result &results);
void print_results (const dev_result &results);

#endif //CULIB_RESULTS_H
