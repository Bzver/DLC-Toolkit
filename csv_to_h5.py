from core.io import csv_op

if __name__ == "__main__":
    csv_file = "CollectedData_bezver"
    project_dir = "D:/Project/DLC-Models/NTD/labeled-data/20250626D-340"
    if csv_op.csv_to_h5(project_dir=project_dir, multi_animal=True,scorer="bezver",csv_name=csv_file):
        print("Success.")