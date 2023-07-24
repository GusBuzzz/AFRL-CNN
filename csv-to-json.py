import csv
import json

def convert_csv_to_json(csv_file):
    json_data = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            json_entry = {
                "FIELD1": int(row["FIELD1"]),
                "filename": row["filename"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "class": row["class"],
                "xmin": row["xmin"],
                "ymin": row["ymin"],
                "xmax": row["xmax"],
                "ymax": row["ymax"],
                "rotated_degrees": int(row["rotated_degrees"]),
                "rotated_xmin": row["rotated_xmin"],
                "rotated_ymin": row["rotated_ymin"],
                "rotated_xmax": row["rotated_xmax"],
                "rotated_ymax": row["rotated_ymax"],
                "object_id": row["object_id"]
            }
            json_data.append(json_entry)
    return json_data

# Replace "input_file.csv" with the path to your CSV file containing the data
csv_file = "train_labels.csv"
json_data = convert_csv_to_json(csv_file)

# Print the converted JSON data
print(json.dumps(json_data, indent=4))
