import os
import numpy as np
import argparse
from tqdm import tqdm

def normalize_pc(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance

    return points

def process_tractors_and_combines(root, num_points):
    # Load data
    train_path = os.path.join(root, 'dataset/sequences/00')
    train_data = [os.path.join(train_path, 'points', f) for f in os.listdir(os.path.join(train_path, 'points'))]
    train_labels = [os.path.join(train_path, 'labels', f) for f in os.listdir(os.path.join(train_path, 'labels'))]
    
    validate_path = os.path.join(root, 'dataset/sequences/01')
    validate_data = [os.path.join(validate_path, 'points', f) for f in os.listdir(os.path.join(validate_path, 'points'))]
    validate_labels = [os.path.join(validate_path, 'labels', f) for f in os.listdir(os.path.join(validate_path, 'labels'))]
    
    test_path = os.path.join(root, 'dataset/sequences/02')
    test_data = [os.path.join(test_path, 'points', f) for f in os.listdir(os.path.join(test_path, 'points'))]
    test_labels = [os.path.join(test_path, 'labels', f) for f in os.listdir(os.path.join(test_path, 'labels'))]
    
    splits_str = ['train', 'validate', 'test']
    splits_data = [train_data, validate_data, test_data]
    splits_labels = [train_labels, validate_labels, test_labels]
    
    # Process data
    # - Add normalized rgb values to the pointcloud (0.5, 0.5, 0.5)
    # - Add normalized xyz values to the pointcloud (Normalized to the unit sphere -> [-1, 1])
    print(f'Processing data at: "{root}"')
    print(f'Sampling point clouds with {num_points} points...')
    for split_name, split_data, split_label in zip(splits_str, splits_data, splits_labels):
        
        # Pathing of processed data
        save_dir = os.path.join(root, 'processed_dgcnn', f'{split_name}')
        points_dir = os.path.join(save_dir, 'points')
        labels_dir = os.path.join(save_dir, 'labels')
        
        # Create directories
        try: 
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(points_dir, exist_ok=False)
            os.makedirs(labels_dir, exist_ok=False)
        except OSError as e: print(e); exit(1)
        
        # Process each pointcloud
        for data_fn, label_fn in tqdm(zip(split_data, split_label), total=len(split_data), desc=f'Processing {split_name} data'):
            # Load the pointcloud and label
            pointcloud = np.load(data_fn)
            label = np.load(label_fn)
            # Only keep the xyz coordinates
            pointcloud = pointcloud[:, :3]
            
            # Ensure pointcloud size is consistent with num_points
            # If the number of points in the data is less than `num_points`, sample with replacement for missing points
            if pointcloud.shape[0] < num_points:
                # Use all points first
                full_choice = np.arange(pointcloud.shape[0])
                
                # Randomly sample additional points to make up the difference
                additional_choice = np.random.choice(pointcloud.shape[0], num_points - pointcloud.shape[0], replace=True)
                
                # Combine the indices
                choice = np.concatenate([full_choice, additional_choice])

            # If the number of points in the data is greater than `num_points`, sample without replacement
            else: 
                choice = np.random.choice(pointcloud.shape[0], num_points, replace=False)
            
            pointcloud = pointcloud[choice, :]
            label = label[choice]
            
            # Create normalized rgb channels
            normalized_rgb = np.full_like(pointcloud, 0.5, dtype=np.float32)
            
            # Create normalized xyz channels
            normalized_pc = normalize_pc(pointcloud)
            
            # Concatenate the normalized rgb and xyz channels
            pointcloud_processed = np.concatenate((pointcloud, normalized_rgb, normalized_pc), axis=1)
            
            # Convert the pointcloud to float16
            pointcloud_processed = pointcloud_processed.astype(np.float16)
            
            # Convert the label to int8
            label = label.round().astype(np.int8)
            
            
            # Save the processed pointcloud and label
            np.save(os.path.join(save_dir, 'points', os.path.basename(data_fn)), pointcloud_processed)
            np.save(os.path.join(save_dir, 'labels', os.path.basename(label_fn)), label)
    
    # Save the number of points sampled
    with open(os.path.join(root, 'processed_dgcnn', 'num_points.txt'), 'w') as file:
        file.write(str(num_points))

    print('Processing complete!')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Tractors and Combines dataset')
    parser.add_argument('--root', type=str, default='data/tractors_and_combines_synth', 
                        help='Path to the root directory of the dataset')
    
    parser.add_argument('--num_points', type=int, default=100_000, 
                        help='Number of points to sample from the pointcloud')
    
    args = parser.parse_args()
    
    # Process the dataset
    process_tractors_and_combines(root=args.root, num_points=args.num_points)