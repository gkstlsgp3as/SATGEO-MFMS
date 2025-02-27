
import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    data = []
    # csv file open
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        # read first row as a dictionary
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    
    # json file save
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)
        
