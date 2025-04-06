import os
import csv

def get_video_data(folder_path, tag):
    video_data = []
    for idx, video_name in enumerate(os.listdir(folder_path)):
        video_path = os.path.join(folder_path, video_name)
        if os.path.isfile(video_path): 
            video_data.append((idx, video_path, tag))
    return video_data

def create_csv_file(output_csv, data):
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