from preprocess_data import split_data, tokenize_dataset
import yaml
import os
import sys
import pandas as pd


def add_data(param_yaml_path: str):
    with open(param_yaml_path) as f:
        params_yaml = yaml.safe_load(f)

    data_path = params_yaml["data_source"]["local_path"]
    data_path = params_yaml["data_source"]["s3_path"]
    # os.makedirs(params_yaml["split"]["dir"], exist_ok=True)

    os.makedirs(os.path.join("/app", params_yaml["split"]["dir"]), exist_ok=True)

    h5_total_file_path = os.path.join(
        "/app", params_yaml["split"]["dir"], params_yaml["split"]["total_file"]
    )
    h5_train_file_path = os.path.join(
        "/app", params_yaml["split"]["dir"], params_yaml["split"]["train_file"]
    )
    h5_val_file_path = os.path.join(
        "/app", params_yaml["split"]["dir"], params_yaml["split"]["val_file"]
    )
    h5_test_file_path = os.path.join(
        "/app", params_yaml["split"]["dir"], params_yaml["split"]["test_file"]
    )

    features, encoded_labels = tokenize_dataset(data_path)

    total_x, total_y, train_x, train_y, valid_x, valid_y, test_x, test_y = split_data(
        features,
        encoded_labels,
        trim_step=params_yaml["split"]["trim_step"],
        split_frac=params_yaml["split"]["split_ratio"],
        random_state=params_yaml["base"]["random_state"],
    )

    # Convert test_x to DataFrame
    df_x = pd.DataFrame(total_x, columns=["feature"])
    df_y = pd.DataFrame(total_y, columns=["label"])

    df = pd.concat([df_x, df_y], axis=1)
    df.to_csv(h5_total_file_path, index=False)

    # Convert test_x to DataFrame
    df_x = pd.DataFrame(test_x, columns=["feature"])
    df_y = pd.DataFrame(test_y, columns=["label"])

    df = pd.concat([df_x, df_y], axis=1)
    df.to_csv(h5_test_file_path, index=False)
    #############################################

    df_x = pd.DataFrame(valid_x, columns=["feature"])
    df_y = pd.DataFrame(valid_y, columns=["label"])

    df = pd.concat([df_x, df_y], axis=1)
    df.to_csv(h5_val_file_path, index=False)
    #############################################

    df_x = pd.DataFrame(train_x, columns=["feature"])
    df_y = pd.DataFrame(train_y, columns=["label"])

    df = pd.concat([df_x, df_y], axis=1)
    df.to_csv(h5_train_file_path, index=False)


if __name__ == "__main__":
    # print( sys.argv )

    if len(sys.argv) != 2:
        raise Exception("Usage: python prepare_dataset.py <path to params.yaml>")
        sys.exit(1)

    add_data(sys.argv[1])
