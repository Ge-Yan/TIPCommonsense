import csv
import os


def read_dataset(file_path, read_concept, num_concept: int):
    dataset = []

    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=',')

        # 跳过第一行（列名）
        # header = next(csv_reader)
        for row in csv_reader:

            FalseSent = row[1]
            if read_concept:
                concepts = row[2:]
                if len(concepts) > num_concept:
                    FalseSent = FalseSent+"," + concepts.pop(0)
                data_entry = {
                    'id': row[0],
                    'FalseSent': FalseSent,
                    'concepts': concepts
                }

                # data_entry = {
                #     'id': int(row[0]),
                #     'FalseSent': FalseSent,
                #     'concepts': concepts
                # }
            else:
                data_entry = {
                    'id': row[0],
                    'FalseSent': FalseSent,
                }
            dataset.append(data_entry)
    return dataset


import csv


def write_to_csv(file_name, data, path=''):
    """
    写入数据到CSV文件

    参数:
    - file_name: 要写入的CSV文件名
    - data: 包含字典或元组的列表，每个字典/元组代表一行数据
    - path: 文件路径，可选参数，默认为空字符串

    示例:
    write_to_csv('output.csv', [{'id': 1, 'content': 'Hello'}, {'id': 2, 'content': 'World'}], path='path/to/')
    """

    # if path is not None:  # make save_path dir
    #     if not os.path.exists(path):
    #         os.makedirs(path, exist_ok=True)
    # 确保文件名以'.csv'结尾
    if not file_name.endswith('.csv'):
        file_name += '.csv'

    # 确保路径存在
    # full_path = os.path.join(path, file_name)
    # os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # 写入数据到CSV文件
    with open(file_name, 'w', newline='') as csvfile:
        # 获取列名，假设数据的第一项是字典
        fieldnames = list(data[0].keys()) if data else []

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入列名
        writer.writeheader()

        # 逐行写入数据
        for row in data:
            writer.writerow(row)
