import os
from core.io import csv_op

if __name__ == "__main__":
    csv_file = "CollectedData_bezver"
    project_dir = "D:/Project/DLC-Models/NTD/labeled-data/20250626D-340"
    csv_path=os.path.join(project_dir, csv_file)
    try:
        csv_op.csv_to_h5(csv_path=csv_path, multi_animal=True,scorer="bezver")
        print("Success")
    except Exception as e:
        print(f"Exception: {e}")