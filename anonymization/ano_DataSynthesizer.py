from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import display_bayesian_network

import argparse
import datetime
import pickle
import pandas as pd
import numpy as np
import builtins
builtins.np = np


def parse_args():
    parser = argparse.ArgumentParser(description="DataSynthesizer anonymization script")
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("-o", "--out", required=True, help="Output CSV file path")
    parser.add_argument("--epsilon", type=float, required=True, help="Epsilon value for differential privacy")
    parser.add_argument("--k", type=int, required=True, help="Degree of Bayesian network")
    parser.add_argument("--mode", type=str, required=True, choices=['correlated_attribute_mode', 'independent_attribute_mode', 'random_mode'], help='Synthesis mode')
    parser.add_argument("--num_tuples", type=int, default=10000, help="Number of tuples to generate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--desc", default=None, help="(Optional) Output path for description file")
    parser.add_argument("--edges", default=None, help="(Optional) Output path for edges pickle file")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    epsilon = args.epsilon
    k = args.k
    input_csv = args.input_csv
    output_csv = args.out
    mode = args.mode
    num_tuples_to_generate = args.num_tuples
    seed = args.seed

    # description/edgesファイルのパス自動生成（指定なければ）
    desc_file = args.desc or (output_csv + f".desc_e{epsilon}_k{k}.json")
    edges_file = args.edges or (output_csv + f".edges_e{epsilon}_k{k}.pkl")

    threshold_value = 21
    categorical_attributes = {'ZIP-code': True, 'Gender': True, 'Age': True, 'Occupation': True}
    candidate_keys = {'Name': True}

    describer = DataDescriber(category_threshold=threshold_value)

    starttime = datetime.datetime.now()

    if mode == "correlated_attribute_mode":
        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=input_csv,
            epsilon=epsilon,
            k=k,
            attribute_to_is_categorical=categorical_attributes,
            attribute_to_is_candidate_key=candidate_keys,
            seed=seed
        )
        display_bayesian_network(describer.bayesian_network)
    elif mode == "independent_attribute_mode":
        describer.describe_dataset_in_independent_attribute_mode(
            dataset_file=input_csv,
            epsilon=epsilon,
            attribute_to_is_categorical=categorical_attributes,
            attribute_to_is_candidate_key=candidate_keys,
            seed=seed
        )
    elif mode == "random_mode":
        describer.describe_dataset_in_random_mode(
            dataset_file=input_csv,
            attribute_to_is_categorical=categorical_attributes,
            attribute_to_is_candidate_key=candidate_keys,
            seed=seed
        )
    else:
        raise Exception('Mode Error!')

    describer.save_dataset_description_to_file(desc_file)

    if mode == "correlated_attribute_mode":
        bn = describer.bayesian_network
        temp = {}
        for child, parents in bn:
            temp[str(child)] = parents
        edges = []
        for child, parents in temp.items():
            for parent in parents:
                edges.append((parent, child))
        with open(edges_file, "wb") as f2:
            pickle.dump(edges, f2)

    generator = DataGenerator()
    if mode == "correlated_attribute_mode":
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, desc_file)
    elif mode == "independent_attribute_mode":
        generator.generate_dataset_in_independent_mode(num_tuples_to_generate, desc_file)
    elif mode == "random_mode":
        generator.generate_dataset_in_random_mode(num_tuples_to_generate, desc_file)
    else:
        raise Exception('Mode Error!')

    generator.save_synthetic_data(output_csv)

    endtime = datetime.datetime.now()
    timedelta = endtime - starttime
    print(f"Elapsed: {timedelta}")

