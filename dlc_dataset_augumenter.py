import os
import sys

import numpy as np
import cv2

import tkinter as tk
from PIL import Image, ImageTk

def get_eligible_dataset(base_folder):

    if not os.path.isdir(base_folder):
        print(f"Error: DLC dataset folder not found at {base_folder}", file=sys.stderr)
        return []
    dataset_folder = []
    for root, dirs, files in os.walk(base_folder):
        dirs[:] = [d for d in dirs if not d.endswith("_labeled") and not d.endswith("_augmented")] 
        if any(f.startswith("CollectedData_") and f.endswith(".h5") for f in files):
            dataset_folder.append(root)
    return dataset_folder

def get_folder_image(folder_path, suffix=".png", prefix=None):

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} does not exist", file=sys.stderr)
        return []
    image_list = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(suffix):
            if prefix is None or file_name.lower().startswith(prefix.lower()):
                image_list.append(os.path.join(folder_path,file_name))
    return image_list

def calculate_histogram_tones(img_data, hist_bins=8, hist_range=[0, 256]):

    if isinstance(img_data, str):
        if not os.path.isfile(img_data):
            print(f"Error: Image not found at {img_data}", file=sys.stderr)
            return False
        img_bgr = cv2.imread(img_data)
    elif isinstance(img_data, np.ndarray):
        img_bgr = img_data
    else:
        print("Error: img_data must be a file path or a NumPy array.", file=sys.stderr)
        return False
    
    ref_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hist_L = cv2.calcHist([ref_lab], [0], None, [hist_bins], hist_range)
    hist_A = cv2.calcHist([ref_lab], [1], None, [hist_bins], hist_range)
    hist_B = cv2.calcHist([ref_lab], [2], None, [hist_bins], hist_range)
    cv2.normalize(hist_L, hist_L, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_A, hist_A, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_B, hist_B, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return (hist_L, hist_A, hist_B)

def compare_tones(img_1_data, img_2_data, image_1_lab=None, image_2_lab=None):

    hist_1_L, hist_1_A, hist_1_B = image_1_lab if image_1_lab else calculate_histogram_tones(img_1_data)
    hist_2_L, hist_2_A, hist_2_B = image_2_lab if image_2_lab else calculate_histogram_tones(img_2_data)

    dist_L = cv2.compareHist(hist_1_L, hist_2_L, cv2.HISTCMP_BHATTACHARYYA)
    dist_A = cv2.compareHist(hist_1_A, hist_2_A, cv2.HISTCMP_BHATTACHARYYA)
    dist_B = cv2.compareHist(hist_1_B, hist_2_B, cv2.HISTCMP_BHATTACHARYYA)

    return (dist_L, dist_A, dist_B)

def interactive_image_adjustment(target_path, ref_path, initial_slider_values=None):

    root = tk.Tk()
    root.title("Image Tone Augmentation GUI")

    max_display_width = 800
    max_display_height = 600

    def center_window(window):
        window.update_idletasks()
        x = (window.winfo_screenwidth() // 2) - max_display_width
        y = (window.winfo_screenheight() // 2) - max_display_height
        window.geometry(f'{max_display_width*2+150}x{max_display_height+125}+{x}+{y}')

    center_window(root)

    # Load images
    original_target_img_bgr = cv2.imread(target_path)
    original_target_img_lab = cv2.cvtColor(original_target_img_bgr, cv2.COLOR_BGR2LAB)
    ref_img_bgr = cv2.imread(ref_path)

    # Resize images to fit window, maintaining aspect ratio


    def resize_image(img_bgr, max_width, max_height):
        h, w = img_bgr.shape[:2]
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img_bgr

    display_ref_img_bgr = resize_image(ref_img_bgr, max_display_width, max_display_height)

    # Convert to PIL PhotoImage for Tkinter display
    def cv2_to_tk(cv2_img_bgr):
        img_rgb = cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        return ImageTk.PhotoImage(pil_img)

    # Image display frames
    image_frame = tk.Frame(root)
    image_frame.pack(side=tk.TOP, padx=10, pady=10)

    target_label = tk.Label(image_frame)
    target_label.pack(side=tk.LEFT, padx=5)

    ref_label = tk.Label(image_frame)
    ref_label.pack(side=tk.RIGHT, padx=5)

    # Sliders for adjustment
    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.BOTTOM, pady=10)

    # Initial values for sliders
    if initial_slider_values is None:
        initial_slider_values = {'brightness': 0, 'contrast': 1.0, 'l': 0, 'a': 0, 'b': 0}

    brightness_val = tk.DoubleVar(value=initial_slider_values['brightness'])
    contrast_val = tk.DoubleVar(value=initial_slider_values['contrast'])
    l_val = tk.DoubleVar(value=initial_slider_values['l']) # L channel adjustment
    a_val = tk.DoubleVar(value=initial_slider_values['a']) # A channel adjustment
    b_val = tk.DoubleVar(value=initial_slider_values['b']) # B channel adjustment

    adjusted_lab_image = original_target_img_lab.copy()

    def update_image(*args):
        nonlocal adjusted_lab_image
        current_target_lab = original_target_img_lab.copy()

        # Apply brightness and contrast in BGR space for simplicity
        # Convert back to BGR, apply, then convert to LAB for L,A,B adjustments
        current_target_bgr = cv2.cvtColor(current_target_lab, cv2.COLOR_LAB2BGR)
        
        # Brightness and Contrast
        alpha = contrast_val.get() # Contrast control (1.0-3.0)
        beta = brightness_val.get() # Brightness control (0-100)
        current_target_bgr = cv2.convertScaleAbs(current_target_bgr, alpha=alpha, beta=beta)
        
        current_target_lab = cv2.cvtColor(current_target_bgr, cv2.COLOR_BGR2LAB)

        # Apply L, A, B adjustments
        l_channel = current_target_lab[:,:,0].astype(np.float32) + l_val.get()
        a_channel = current_target_lab[:,:,1].astype(np.float32) + a_val.get()
        b_channel = current_target_lab[:,:,2].astype(np.float32) + b_val.get()

        # Clip values to valid LAB ranges
        l_channel = np.clip(l_channel, 0, 255)
        a_channel = np.clip(a_channel, 0, 255)
        b_channel = np.clip(b_channel, 0, 255)

        adjusted_lab_image = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.uint8)
        
        # Convert back to BGR for display
        display_img_bgr = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2BGR)
        display_img_bgr = resize_image(display_img_bgr, max_display_width, max_display_height)
        
        tk_img = cv2_to_tk(display_img_bgr)
        target_label.config(image=tk_img)
        target_label.image = tk_img

        # Perform tone comparison and update labels
        adjusted_img_for_comparison = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2BGR)
        
        dist_L, dist_A, dist_B = compare_tones(ref_img_bgr, adjusted_img_for_comparison)
        avg_dist = (dist_L + dist_A + dist_B) / 3

        l_dist_label.config(text=f"L Distance: {dist_L:.4f}")
        a_dist_label.config(text=f"A Distance: {dist_A:.4f}")
        b_dist_label.config(text=f"B Distance: {dist_B:.4f}")
        avg_dist_label.config(text=f"Average Distance: {avg_dist:.4f}")

    # Create sliders
    slider_frame = tk.LabelFrame(control_frame, text="CONTROL SLIDERS", padx=10, pady=10)
    slider_frame.pack(side=tk.LEFT, padx=10)

    btgt_scale = tk.Scale(slider_frame, label="Brightness", from_=-100, to=100, orient=tk.HORIZONTAL, length=200, variable=brightness_val, command=update_image)
    btgt_scale.pack(side=tk.LEFT, padx=5)
    ct_scale = tk.Scale(slider_frame, label="Contrast", from_=0.1, to=3.0, resolution=0.01, orient=tk.HORIZONTAL, length=200, variable=contrast_val, command=update_image)
    ct_scale.pack(side=tk.LEFT, padx=5)
    l_scale = tk.Scale(slider_frame, label="L (Lightness)", from_=-100, to=100, orient=tk.HORIZONTAL, length=200, variable=l_val, command=update_image)
    l_scale.pack(side=tk.LEFT, padx=5)
    a_scale = tk.Scale(slider_frame, label="A (Green-Red)", from_=-100, to=100, orient=tk.HORIZONTAL, length=200, variable=a_val, command=update_image)
    a_scale.pack(side=tk.LEFT, padx=5)
    b_scale = tk.Scale(slider_frame, label="B (Blue-Yellow)", from_=-100, to=100, orient=tk.HORIZONTAL, length=200, variable=b_val, command=update_image)
    b_scale.pack(side=tk.LEFT, padx=5)

    # Labels for tone comparison results
    compare_frame = tk.LabelFrame(control_frame, text="COMPARED METRICS", padx=10, pady=10)
    compare_frame.pack(side=tk.LEFT, padx=10)

    l_dist_label = tk.Label(compare_frame, text="L Distance: N/A")
    l_dist_label.pack(pady=2)
    a_dist_label = tk.Label(compare_frame, text="A Distance: N/A")
    a_dist_label.pack(pady=2)
    b_dist_label = tk.Label(compare_frame, text="B Distance: N/A")
    b_dist_label.pack(pady=2)
    avg_dist_label = tk.Label(compare_frame, text="Average Distance: N/A")
    avg_dist_label.pack(pady=2)

    # Save button
    save_flag = False
    auto_apply_flag = False

    def save_and_close():
        nonlocal save_flag
        save_flag = True
        root.destroy()

    def auto_apply_and_close():
        nonlocal save_flag, auto_apply_flag
        save_flag = True
        auto_apply_flag = True
        root.destroy()

    def on_closing():
        sys.exit(1)
        root.destroy()

    save_button = tk.Button(compare_frame, text="Apply & Close", command=save_and_close)
    save_button.pack(side=tk.BOTTOM, pady=2)

    auto_apply_button = tk.Button(compare_frame, text="Auto Apply to Rest", command=auto_apply_and_close)
    auto_apply_button.pack(side=tk.BOTTOM, pady=2)

    root.protocol("WM_DELETE_WINDOW", on_closing)

    ref_tk_img = cv2_to_tk(display_ref_img_bgr)
    ref_label.config(image=ref_tk_img)
    ref_label.image = ref_tk_img
    update_image()

    root.mainloop()

    current_slider_values = {
        'brightness': brightness_val.get(),
        'contrast': contrast_val.get(),
        'l': l_val.get(),
        'a': a_val.get(),
        'b': b_val.get()
    }

    if save_flag:
        return adjusted_lab_image, current_slider_values, auto_apply_flag
    else:
        return None, None, False

def match_tones(ref_paths, target_paths, output_dir, threshold=0.1):

    if isinstance(ref_paths,list):
        ref_lab = (0,0,0)
        num_ref = 0
        for ref_path in ref_paths:
            if not os.path.isfile(ref_path):
                print(f"Error: Reference image not found at {ref_path}", file=sys.stderr)
            else:
                hist_L, hist_A, hist_B = calculate_histogram_tones(ref_path)
                if hist_L is not False: # Check if histogram calculation was successful
                    ref_lab = (ref_lab[0] + hist_L, ref_lab[1] + hist_A, ref_lab[2] + hist_B)
                    num_ref += 1
        
        if num_ref == 0:
            print("Error: No valid reference images found.", file=sys.stderr)
            return False # Or raise an exception, depending on desired error handling
        
        ref_lab = (ref_lab[0] / num_ref, ref_lab[1] / num_ref, ref_lab[2] / num_ref)
        ref_path = ref_paths[0] # Keep the first path for comparison function that needs a path

    elif os.path.isfile(ref_paths):
        ref_path = ref_paths
        ref_lab = calculate_histogram_tones(ref_path)
        if ref_lab is False: # Handle case where single ref image is not found
            print(f"Error: Reference image not found at {ref_path}", file=sys.stderr)
            return False
    else:
        print(f"Error: Reference image not found at {ref_paths}", file=sys.stderr)
        return False
    
    skipped_img = []

    for target_path in target_paths:
        if not os.path.isfile(target_path):
            print(f"Error: Target image not found at {target_path}", file=sys.stderr)
            skipped_img.append(target_path)
        else:
            dist_L, dist_A, dist_B = compare_tones(ref_path, target_path, image_1_lab=ref_lab)
            avg_dist = (dist_L + dist_A + dist_B) / 3

            if avg_dist < threshold:
                print(f"Skipping {target_path} as tones are already similar (average distance: {avg_dist:.4f})", file=sys.stderr)
                skipped_img.append(target_path)
            else:
                # Initialize last_slider_values and auto_apply_active for the first image
                if 'last_slider_values' not in locals():
                    last_slider_values = {'brightness': 0, 'contrast': 1.0, 'l': 0, 'a': 0, 'b': 0}
                    auto_apply_active = False # Initialize auto_apply_active

                if auto_apply_active:
                    print(f"Auto-applying adjustments to {target_path}...")
                    target_img_bgr = cv2.imread(target_path)
                    
                    # Apply brightness and contrast in BGR space
                    target_img_bgr_adjusted = cv2.convertScaleAbs(target_img_bgr,
                                                                  alpha=last_slider_values['contrast'],
                                                                  beta=last_slider_values['brightness'])
                    target_img_lab_adjusted = cv2.cvtColor(target_img_bgr_adjusted, cv2.COLOR_BGR2LAB)

                    # Apply L, A, B adjustments
                    l_channel = target_img_lab_adjusted[:,:,0].astype(np.float32) + last_slider_values['l']
                    a_channel = target_img_lab_adjusted[:,:,1].astype(np.float32) + last_slider_values['a']
                    b_channel = target_img_lab_adjusted[:,:,2].astype(np.float32) + last_slider_values['b']

                    l_channel = np.clip(l_channel, 0, 255)
                    a_channel = np.clip(a_channel, 0, 255)
                    b_channel = np.clip(b_channel, 0, 255)

                    matched_lab = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.uint8)
                    matched_img = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
                    
                    output_path = f"{output_dir}/{os.path.basename(target_path)}"
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(output_path, matched_img)
                    print(f"Processed and saved {target_path} to {output_path} (auto-applied)")

                else:
                    matched_lab, current_slider_values, auto_apply_active = interactive_image_adjustment(target_path, ref_path, last_slider_values)
                    
                    if matched_lab is not None: # Only save if the user pressed "Apply & Close" or "Auto Apply to Rest"
                        matched_img = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
                        
                        output_path = f"{output_dir}/{os.path.basename(target_path)}"
                        os.makedirs(output_dir, exist_ok=True)
                        cv2.imwrite(output_path, matched_img)
                        print(f"Processed and saved {target_path} to {output_path}")
                        last_slider_values = current_slider_values # Update last used values
                    else:
                        print(f"Skipping {target_path} as adjustment was cancelled by user.", file=sys.stderr)

    print("Processing complete.")
    if skipped_img:
        print(f"Skipped target images:\n{skipped_img}")
        
if __name__ == "__main__":
    dlcDatasetFolder = "D:/Project/DLC-Models/NTD/labeled-data"
    referenceFolder = "D:/DGH/Data/Videos/2025-06-26 7D Marathon/video_previews/sampled_frames"
    additionalRefFolder = ["D:/DGH/Data/Videos/2025-06-26 7D Marathon/video_previews/sampled_frames_2"]

    dataset_folders = get_eligible_dataset(dlcDatasetFolder)
    if not dataset_folders:
        print("No eligible DLC datasets found. Exiting.", file=sys.stderr)
        sys.exit(1)
    dataset_folder = dataset_folders[0]

    ref_images = get_folder_image(referenceFolder, prefix="frame_")
    if additionalRefFolder:
        ref_images_extra = []
        for folder in additionalRefFolder:
            ref_images_extra += get_folder_image(folder, prefix="frame_")
        ref_images += ref_images_extra

    target_images = get_folder_image(dataset_folder, prefix="img")
    output_directory = dataset_folder+"_augmented"
    os.makedirs(output_directory, exist_ok=True)

    match_tones(ref_images, target_images, output_directory)