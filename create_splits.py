import pandas as pd


videos_dir = 'data/images'
csv_data = pd.read_csv('/data/annotations.csv')
video_ids = csv_data['video_id'].unique()
videos = []
annotations = []
bboxes = []
# Iterate over all videos
for id in video_ids:
    video_data = csv_data[csv_data['video_id'] == id]['subject_id'].unique()
    # Iterate over all birds in the video
    for subject_id in video_data:
        # Get activity boundaries for each bird actvity
        video_bird_data = csv_data[(csv_data['video_id'] == id) & (csv_data['subject_id'] == subject_id)]
        video_bird_data = video_bird_data.sort_values(by='frame_id')

        start_frame = video_bird_data.iloc[0]['frame_id']
        current_activity = video_bird_data.iloc[0]['activity_id']
        counter = 0
        for i in range(1, len(video_bird_data)):
            if video_bird_data.iloc[i]['activity_id'] != current_activity:
                end_frame = video_bird_data.iloc[i-1]['frame_id']
                videos.append((id, subject_id, counter))
                counter += 1
                annotations.append((start_frame, end_frame, current_activity))
                start_frame = video_bird_data.iloc[i]['frame_id']
                current_activity = video_bird_data.iloc[i]['activity_id']

        # Get bounding boxes for each bird of the video
        bboxes.append([])
        for i in range(len(video_bird_data)):
            bboxes[-1].append((video_bird_data.iloc[i]['xtl'], video_bird_data.iloc[i]['ytl'], video_bird_data.iloc[i]['xbr'], video_bird_data.iloc[i]['ybr']))

