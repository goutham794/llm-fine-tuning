from dataset import Dataset, DatasetDict, load_dataset


def get_dataset(csv_path: str, max_length: int):
    dataset = load_dataset("csv", data_files=csv_path)

    # Split the dataset into training and evaluation sets
    train_test_split = dataset["train"].train_test_split(test_size=0.15)
    dataset = DatasetDict(
        {"train": train_test_split["train"], "eval": train_test_split["test"]}
    )
