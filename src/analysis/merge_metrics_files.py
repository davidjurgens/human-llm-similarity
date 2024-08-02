import argparse
import glob
import pandas as pd



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="The directory with metrics jsonl files",
                        required=True)
    parser.add_argument("--output_path", type=str, help="Name of the file to save the combined metrics",
                        required=True)

    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = args.output_path

    dframe = pd.DataFrame()
    for fname in glob.glob(f"{input_dir}/*.jsonl"):
        curr_dframe = pd.read_json(fname, orient='records', lines=True)
        cols_to_use = curr_dframe.columns.difference(dframe.columns)
        dframe = pd.merge(dframe, curr_dframe[cols_to_use], left_index=True, right_index=True, how='outer')

    dframe.to_json(output_path, orient='records', lines=True)
