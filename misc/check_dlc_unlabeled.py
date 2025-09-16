import os

dir = "D:/Project/DLC-Models/NTD/labeled-data"

folders = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]

missing = []
img_count = 0

for folder in folders:
    if not f"{folder}_labeled" in folders:
        continue

    paired_folder = f"{folder}_labeled"
    for folder_file in os.listdir(folder):
        if not folder_file.endswith(".png"):
            continue

        img_count += 1
        if f"{folder_file[:-4]}_individual.png" not in os.listdir(paired_folder):
            missing.append(os.path.join(folder, folder_file))

if missing:
    print(f"{len(missing)} pictures are somehow not registered by DLC.")
    print("And they will not be used as training data. Unregistered:")
    print('\n'.join(missing))
else:
    print("No missing.")
    print(img_count)