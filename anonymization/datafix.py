import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="DataSynthesizer anonymization script")
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("-o", "--out", required=True, help="Output CSV file path")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    input_csv = args.input_csv
    output_csv = args.out
    mode = args.mode
    num_tuples_to_generate = args.num_tuples
    seed = args.seed

    # description/edgesファイルのパス自動生成（指定なければ）
    desc_file = args.desc or (output_csv + f".desc_e{epsilon}_k{k}.json")
    edges_file = args.edges or (output_csv + f".edges_e{epsilon}_k{k}.pkl")