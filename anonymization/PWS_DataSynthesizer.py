from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import sys
import pandas as pd
import datetime
import pickle

if __name__ == '__main__':
    param = sys.argv
    tempe = param[1]
    temp2e = (tempe[1:-1].split(","))
    e = [float(s) for s in temp2e]
    tempk = param[2]
    temp2k = (tempk[1:-1].split(","))
    k = [int(s) for s in temp2k]
    input = param[3]
    output = param[4]
    id = param[5]
    mode = param[6]

    # e = [20,15,10,5,3,2]
    # k = [4]
    # input = "./in"
    # output = "./out"
    # id = 17
    # mode = "correlated_attribute_mode"

    for epsilon in (e):
    # for epsilon in ([20,15,10,5,3,2]):
    # for epsilon in (2,2):
        for degree_of_bayesian_network in (k):
        # for degree_of_bayesian_network in (3,3):
            file = open(f'{output}/{mode}/description_full_e{epsilon}_BN{degree_of_bayesian_network}.txt', "w")
            file.close()
            f = open(f'{output}/{mode}/description_full_e{epsilon}_BN{degree_of_bayesian_network}.txt', "a")
            print("epsilon=" + format(epsilon), file=f)
            print("k=" + format(degree_of_bayesian_network), file=f)
            print("epsilon=" + format(epsilon))
            print("k=" + format(degree_of_bayesian_network))
            # set starttime
            starttime = datetime.datetime.now()

            # input dataset
            input_data = f'{input}/B{id}.csv'
            # location of two output files
            # mode = 'correlated_attribute_mode'
            description_file = f'{output}/{mode}/description_full_e{epsilon}_BN{degree_of_bayesian_network}.json'
            synthetic_data = f'{output}/{mode}/sythetic_data_full_e{epsilon}_BN{degree_of_bayesian_network}.csv'

            # An attribute is categorical if its domain size is less than this threshold.
            # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
            threshold_value = 21

            # specify categorical attributes
            categorical_attributes = {'ZIP-code': True, 'Gender': True, 'Age': True, 'Occupation': True}

            # specify which attributes are candidate keys of input dataset.
            candidate_keys = {'Name': True}

            # A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not 
            # change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
            # Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
            # epsilon = 5

            # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
            # degree_of_bayesian_network = 3

            # Number of tuples generated in synthetic dataset.
            num_tuples_to_generate = 10000 # Here 32561 is the same as input dataset, but it can be set to another number.

            describer = DataDescriber(category_threshold=threshold_value)

            if mode == "correlated_attribute_mode":
                describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                                    epsilon=epsilon, 
                                                                    k=degree_of_bayesian_network,
                                                                    attribute_to_is_categorical=categorical_attributes,
                                                                    attribute_to_is_candidate_key=candidate_keys,
                                                                    seed=1)
                display_bayesian_network(describer.bayesian_network)
            elif mode == "independent_attribute_mode":
                describer.describe_dataset_in_independent_attribute_mode(dataset_file=input_data, 
                                                                        epsilon=epsilon, 
                                                                        attribute_to_is_categorical=categorical_attributes,
                                                                        attribute_to_is_candidate_key=candidate_keys,
                                                                        seed=1)
            elif mode == "random_mode":
                describer.describe_dataset_in_random_mode(dataset_file=input_data, 
                                                                        attribute_to_is_categorical=categorical_attributes,
                                                                        attribute_to_is_candidate_key=candidate_keys,
                                                                        seed=1)
            else:
                raise Exception('Mode Error!')
            
            describer.save_dataset_description_to_file(description_file)

            # describer.bayesian_networkは辞書形式で以下の形式で取り出すことができる。
            """
            length = 0
            for child, _ in bn:
                if len(child) > length:
                    length = len(child)

            print('Constructed Bayesian network:')
            for child, parents in bn:
                print("    {0:{width}} has parents {1}.".format(child, parents, width=length))
            """
            if mode == "correlated_attribute_mode":
                bn = describer.bayesian_network
                print('Constructed Bayesian network:')
                temp = {}
                for child, parents in bn:
                    temp[str(child)] = parents

                # エッジリストを生成
                edges = []
                for child, parents in temp.items():
                    for parent in parents:
                        edges.append((parent, child))  # (parent, child) の形式でエッジを追加

                print(edges, file=f)
                with open(f'{output}/{mode}/edges_full_e{epsilon}_BN{degree_of_bayesian_network}.pkl', "wb")as f2:
                    pickle.dump(edges, f2)

            generator = DataGenerator()
            if mode == "correlated_attribute_mode":
                generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
            elif mode == "independent_attribute_mode":
                generator.generate_dataset_in_independent_mode(num_tuples_to_generate, description_file)
            elif mode == "random_mode":
                generator.generate_dataset_in_random_mode(num_tuples_to_generate, description_file)
            else:
                raise Exception('Mode Error!')
            
            generator.save_synthetic_data(synthetic_data)

            # # Read both datasets using Pandas.
            # input_df = pd.read_csv(input_data, skipinitialspace=True)
            # synthetic_df = pd.read_csv(synthetic_data)
            # # Read attribute description from the dataset description file.
            # attribute_description = read_json_file(description_file)['attribute_description']

            # inspector = ModelInspector(input_df, synthetic_df, attribute_description)

            # for attribute in synthetic_df.columns:
            #     inspector.compare_histograms(attribute)

            # inspector.mutual_information_heatmap()

            # set endtime
            endtime = datetime.datetime.now()
            timedelta = endtime - starttime
            print(timedelta, file=f)
            f.close()
            
