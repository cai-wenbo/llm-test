from categories import categories, subcategories
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
            accuracy_dict[row[0]] = float(row[1])

        f.close()


    scores = dict()
    for key in categories:
        category_accuracy = list()
        for subject in subcategories:
            if subcategories[subject][0] in categories[key]:
                category_accuracy.append(accuracy_dict[subject])

        category_accuracy = np.array(category_accuracy)
        scores[key] = np.mean(category_accuracy)

    write_dict(score_path, scores)
