#!/bin/python3

import pandas as pd
import os


def parse_dependent_implementation(result_dir):
    result = {}
    for data_type in os.listdir(result_dir):
        file_path = os.path.join(result_dir, data_type)
        result[data_type] = pd.read_csv(file_path, header=None)
    return result


def parse_implementations(result_dir, is_dependent):
    result = {}
    for implementation in os.listdir(result_dir):
        if is_dependent:
            result[implementation] = parse_dependent_implementation(os.path.join(result_dir, implementation))
    return result


def parse_algorithms(results_dir, is_dependent):
    result = {}
    for algorithm in os.listdir(results_dir):
        algo_path = os.path.join(results_dir, algorithm)
        result[algorithm] = parse_implementations(algo_path, is_dependent)
    return result


def parse_scopes(results_dir, is_dependent):
    result = {}
    for scope in ["warp", "block", "device"]:
        scope_path = os.path.join(results_dir, scope)

        if not os.path.exists(scope_path):
            continue

        result[scope] = parse_algorithms(scope_path, is_dependent)

    return result


def parse_devices(results_dir, is_dependent):
    result = {}
    for device in os.listdir(results_dir):
        device_dir = os.path.join(results_dir, device)
        result[device] = parse_scopes(device_dir, is_dependent)
    return result


def parse(results_dir, is_dependent):
    target_name = "dependent" if is_dependent else "independent"
    results_path = os.path.join(results_dir, target_name)

    if os.path.exists(results_path):
        return parse_devices(results_path, is_dependent)
    return {}


def parse_dependent(results_dir):
    return parse(results_dir, True)


def parse_independent(results_dir):
    return parse(results_dir, False)


def parse_results(results_dir):
    return {"dependent": parse_dependent(results_dir),
            "indepemdent": parse_independent(results_dir)}


def main():
    data_path = "../data"

    for commit in os.listdir(data_path):
        commit_path = os.path.join(data_path, commit)
        result = parse_results(commit_path)
        print(result["dependent"]["GeForce RTX 2080"]["warp"]["reduce"]["culib"]["int"])


if __name__ == '__main__':
    main()
