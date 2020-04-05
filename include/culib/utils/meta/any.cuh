//
// Created by egi on 4/5/20.
//

#ifndef CULIB_ANY_H
#define CULIB_ANY_H

namespace culib
{
namespace utils
{
namespace meta
{

template<typename T, typename... Rest>
struct is_any : std::false_type {};

template<typename T, typename First>
struct is_any<T, First> : std::is_same<T, First> {};

template<typename T, typename First, typename... Rest>
struct is_any<T, First, Rest...> : std::integral_constant<bool, std::is_same<T, First>::value || is_any<T, Rest...>::value>
{};

}
}
}

#endif //CULIB_ANY_H
