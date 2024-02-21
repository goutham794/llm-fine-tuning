from dataset import Dataset, DatasetDict, load_dataset


def format_row(row):
    indices = ast.literal_eval(row['indices'])
    span = row['sentence'][indices[-1]:indices[1]]
    try:
        assert span == row['span']
    except Exception:
        print(row['sentence'])
        print(span)
        print(row['span'])
        print(indices)
        print("next")
    formatted_string = PROMPT_TEMPLATE.format(row['topic'], row['sentence'], span, EOS_TOKEN)
    return {'text' : formatted_string}

def get_data(csv_path, EOS_TOKEN, valid_set_ratio = 0.15):
    data = load_dataset('csv', data_files=csv_path)
    dataset = data.map(format_row, remove_columns=data.column_names['train'])
    # Split the dataset into training and evaluation sets
    train_test_split = dataset['train'].train_test_split(test_size=0.15)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'eval': train_test_split['test']
    })
    return dataset
