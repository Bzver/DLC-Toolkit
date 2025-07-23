import numpy as np
import h5py

def get_vector_from_points(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    return p2 - p1

def get_new_point(start_point, direction_vector, multiplier=0.5):
    start_point = np.asarray(start_point)
    return start_point + multiplier * direction_vector

def keypoiny_duplicator(sleap_file, num_bodypart, head_tail=None, tail_scheme=None, limb_scheme=None):
    """
    Duplicates and fills in missing keypoints in a SLEAP HDF5 file based on specified schemes.

    This function takes a SLEAP HDF5 file, reads the keypoint data, and then
    fills in missing keypoints (NaN values) based on either a tail extension scheme
    or a limb extension scheme. It modifies the 'points' dataset within the HDF5 file.

    Args:
        sleap_file (str): The path to the SLEAP HDF5 file.
        num_bodypart (int): The number of body parts per frame in the SLEAP data.
        head_tail (tuple, optional): A tuple (head_index, tail_index) used to calculate
                                     the body vector for tail extension. Indices are 1-based.
                                     If None, a random direction is used for tail extension.
        tail_scheme (list of tuples, optional): A list of (source_index, new_index) tuples
                                                for tail keypoint duplication. Indices are 1-based.
                                                The new_index keypoint will be extended from the
                                                source_index keypoint along the body vector.
        limb_scheme (list of tuples, optional): A list of (source_index, new_index) tuples
                                                for limb keypoint duplication. Indices are 1-based.
                                                The new_index keypoint will be extended from the
                                                source_index keypoint using a small random vector.
    """
    if tail_scheme is None and limb_scheme is None:
        print("tail_scheme and limb_scheme not provided, no changes will be made.")
        return
    else:
        with h5py.File(sleap_file, "r") as hdf_file:
            points = np.array(list(hdf_file["points"]))

            if tail_scheme is not None:
                if head_tail is None:
                    print("Head_tail index not provided, will use random direction for tail extension.")
                    head_ind, tail_ind = None, None
                else:
                    head_ind = head_tail[0] - 1
                    tail_ind = head_tail[1] - 1
                for i in range(0, points.shape[0], num_bodypart): # Process through each frame
                    # Calculate the body vector for tail extension
                    if head_ind is None:
                        body_vector = np.asarray(np.random.uniform(-100, 100, size=2)) # Randomly choose a direction
                    else:
                        frame_head_ind = i + head_ind
                        frame_tail_ind = i + tail_ind
                        head_coord = points[frame_head_ind][0], points[frame_head_ind][1]
                        tail_coord = points[frame_tail_ind][0], points[frame_tail_ind][1]
                        body_vector = get_vector_from_points(head_coord, tail_coord)
                    for src, new in tail_scheme:
                        src_coord = points[i+src-1][0], points[i+src-1][1]
                        new_coord = points[i+new-1][0], points[i+new-1][1]
                        if np.all(np.isnan(new_coord)): # only process empty points
                            new_coord = get_new_point(src_coord, body_vector)
                        points[i+new-1][0], points[i+new-1][1] = new_coord

            if limb_scheme is not None:
                for k in range(0, points.shape[0], num_bodypart):
                    for src, new in limb_scheme:
                        src_coord = points[k+src-1][0], points[k+src-1][1]
                        random_vector = np.asarray(np.random.uniform(-5, 5, size=2))
                        new_coord = points[k+new-1][0], points[k+new-1][1]
                        if np.all(np.isnan(new_coord)):
                            new_coord = get_new_point(src_coord, random_vector)
                        points[k+new-1][0], points[k+new-1][1] = new_coord

        # Re-open the file in read/write mode to save changes
        with h5py.File(sleap_file, "r+") as hdf_file:
            # Delete the old 'points' dataset
            if "points" in hdf_file:
                del hdf_file["points"]
            # Create a new 'points' dataset with the modified data
            hdf_file.create_dataset("points", data=points)
            print(f"Modified points saved to {sleap_file}")
        
if __name__ == "__main__":
    sleap_file = "D:/Project/Sleap-Models/BTR/labels.NS2000.slp"
    head2tailIndex = (1,4)
    tailExtensiontChart = [(4,11),(11,12)]
    LimbExtensiontChart = [(7,13),(8,14),(9,15),(10,16)]
    keypoiny_duplicator(sleap_file, 16, head_tail=head2tailIndex, tail_scheme=tailExtensiontChart, limb_scheme=LimbExtensiontChart)