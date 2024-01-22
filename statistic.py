from categories import categories
import csv
import numpy as np



def write_dict(table_path, encoding_dict):
    with open(table_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in encoding_dict.items():
            writer.writerow([key,value])
        csv_file.close()


if __name__ == "__main__":
    accuracy_path = "./accuracy.csv"
    score_path = "./scores.csv"
    accuracy_dict = dict()
    with open(accuracy_path, 'r') as f:
        csv_reader = csv.reader(f)
        
        for row in csv_reader:
            accuracy_dict[row[0]] = int(row[1])

        f.close()

    print(accuracy_dict)

    scores = dict()
    for key in categories:
        accuracy_list = np.ndarray([accuracy_dict[subject] for subject in categories[key]])
        score[key] = np.mean(accuracy_list)

    write_dict(score_path, scores)
