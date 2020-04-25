//
// Created by egi on 4/25/20.
//

#include "results.h"
#include "culib/utils/version.cuh"

#include <iostream>
#include <fstream>

#include <experimental/filesystem>


namespace fs = std::experimental::filesystem;

template <typename container_type>
void dump (const container_type &results, const std::string &parent_dir)
{
  for (auto &it: results)
    {
      const std::string dir = parent_dir + "/" + it.first;
      if (!fs::is_directory (dir) && !fs::exists (dir))
        fs::create_directory (dir);
      dump (it.second, dir);
    }
}

template <>
void dump (const clk_result &results, const std::string &dir)
{
  for (auto &it: results)
    {
      std::ofstream ofs (dir + "/" + it.first);
      for (auto &n_clk: it.second)
        ofs << n_clk.first << "," << n_clk.second << "\n";
    }
}

void dump_results (const dev_result &results)
{
  std::string root_dir;
  for (auto &dir: { "data/", CULIB_VERSION, "/dependent" })
    {
      root_dir += dir;
      if (!fs::is_directory (root_dir) && !fs::exists (root_dir))
        fs::create_directory (root_dir);
    }

  dump (results, root_dir);
}

template <typename container_type>
void print (const container_type &results, int scope)
{
  std::string tabs (scope, '\t');

  for (auto &it: results)
    {
      std::cout << tabs << it.first << "\n";
      print (it.second, scope + 1);
    }
}

template <>
void print (const clk_result &results, const int scope)
{
  std::string tabs (scope, '\t');

  for (auto &it: results)
    {
      std::cout << tabs << it.first << ": ";
      for (auto &n_clk: it.second)
        std::cout << "{ " << n_clk.first << ": " << n_clk.second << " } ";
      std::cout << "\n";
    }
}

void print_results (const dev_result &results)
{
  print (results, 0);
}
