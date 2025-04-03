import os
import csv

def get_video_data(folder_path, tag):
    """
    Get video data from a folder and assign a tag.
    
    Args:
        folder_path (str): Path to the folder containing videos.
        tag (str): Tag to assign to the videos (e.g., 'real' or 'fake').
    
    Returns:
        list: List of tuples containing video index, video path, and tag.
    """
    video_data = []
    for idx, video_name in enumerate(os.listdir(folder_path)):
        video_path = os.path.join(folder_path, video_name)
        if os.path.isfile(video_path): 
            video_data.append((idx, video_path, tag))
    return video_data

def create_csv_file(output_csv, data):
    """
    Create a CSV file with the given data.
    
    Args:
        output_csv (str): Path to the output CSV file.
        data (list): List of tuples containing video data.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Video Name', 'Tag']) 
        writer.writerows(data)

def main():
    train_folder = '/Users/kavyasmac/Desktop/Time Pass/DeepFakeDetection/Data/ModelData/Test'
    output_csv = './CSV Files/test.csv'
    
    real_folder = os.path.join(train_folder, 'real')
    fake_folder = os.path.join(train_folder, 'fake')
    
    real_videos = get_video_data(real_folder, 'real')
    fake_videos = get_video_data(fake_folder, 'fake')
    
    all_videos = real_videos + fake_videos
    create_csv_file(output_csv, all_videos)

if __name__ == "__main__":
    main()