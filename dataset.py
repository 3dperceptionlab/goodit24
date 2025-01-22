from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from PIL import Image
import numpy as np
from torchvision.models.video import R3D_18_Weights, S3D_Weights, Swin3D_T_Weights, MViT_V1_B_Weights, R2Plus1D_18_Weights, Swin3D_B_Weights
from sklearn.model_selection import train_test_split

#class to load annotations from csv
class BehaviorsDataset(Dataset):
    def __init__(self, csv_file, videos_dir, split='train', n_frames = 177, model_name='3dresnet'):
        self.videos_dir = videos_dir
        csv_data = pd.read_csv(csv_file)
        video_ids = csv_data['video_id'].unique()
        self.videos = []
        self.annotations = []
        self.bboxes = []
        self.n_frames = n_frames
        self.model_name = model_name
        
        # Get processing model
        if self.model_name == '3dresnet':
            self.preprocess = R3D_18_Weights.DEFAULT
        elif self.model_name == 'swin':
            self.preprocess = Swin3D_T_Weights.DEFAULT
        elif self.model_name == 's3d':
            self.preprocess = S3D_Weights.DEFAULT
        elif self.model_name == 'mvit':
            self.preprocess = MViT_V1_B_Weights.DEFAULT
        elif self.model_name == 'r2plus':
            self.preprocess = R2Plus1D_18_Weights.DEFAULT
        elif self.model_name == 'swin3d_b':
            self.preprocess = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1

        # Iterate over all videos
        for id in video_ids:
            video_data = csv_data[csv_data['video_id'] == id]['subject_id'].unique()
            # Iterate over all birds in the video
            for subject_id in video_data:
                # Get activity boundaries for each bird actvity
                video_bird_data = csv_data[(csv_data['video_id'] == id) & (csv_data['subject_id'] == subject_id)]
                video_bird_data = video_bird_data.sort_values(by='frame_id')

                frame_ids = []
                start_frame = video_bird_data.iloc[0]['frame_id']
                current_activity = video_bird_data.iloc[0]['activity_id']
                counter = 0
                for i in range(len(video_bird_data)):
                    if video_bird_data.iloc[i]['activity_id'] != current_activity:
                        if len(frame_ids) < 30: # Ignore activities with less than 30 frames
                            break
                        end_frame = video_bird_data.iloc[i-1]['frame_id']
                        self.videos.append((id, subject_id, counter))
                        counter += 1
                        self.annotations.append((frame_ids, current_activity))

                        self.bboxes.append([])
                        video_bird_act_data = video_bird_data[video_bird_data['frame_id'].isin(frame_ids)]
                        for j in range(len(video_bird_act_data)):
                            self.bboxes[-1].append((video_bird_act_data.iloc[j]['xtl'], video_bird_act_data.iloc[j]['ytl'], video_bird_act_data.iloc[j]['xbr'], video_bird_act_data.iloc[j]['ybr'], video_bird_act_data.iloc[j]['frame_id']))

                        frame_ids = []
                        start_frame = video_bird_data.iloc[i]['frame_id']
                        current_activity = video_bird_data.iloc[i]['activity_id']
                    else:
                        frame_ids.append(video_bird_data.iloc[i]['frame_id'])

        # Split train-test data
        videos_train, videos_test, annotations_train, annotations_test, bboxes_train, bboxes_test = train_test_split(self.videos, self.annotations, self.bboxes, test_size=0.2, random_state=42, stratify=[x[1] for x in self.annotations])


        # Duplicate fly actions in test set
        fly_idx = [i for i, x in enumerate(annotations_test) if x[1] == 2]
        assert len(fly_idx) == 1, 'There should be only one fly action in the test set'
        fly_idx = fly_idx[0]
        videos_test.append(videos_test[fly_idx])
        annotations_test.append(annotations_test[fly_idx])
        bboxes_test.append(bboxes_test[fly_idx])

        # Split test-valid data
        videos_test, videos_valid, annotations_test, annotations_valid, bboxes_test, bboxes_valid = train_test_split(videos_test, annotations_test, bboxes_test, test_size=0.5, random_state=42, stratify=[x[1] for x in annotations_test])

        # Assign split
        if split == 'train':
            self.videos = videos_train
            self.annotations = annotations_train
            self.bboxes = bboxes_train
        elif split == 'test':
            self.videos = videos_test
            self.annotations = annotations_test
            self.bboxes = bboxes_test
        else:
            self.videos = videos_valid
            self.annotations = annotations_valid
            self.bboxes = bboxes_valid
        print(len(self.videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # Get data
        video_id, subject_id, _ = self.videos[idx]
        indexes, activity = self.annotations[idx]
        bboxes = self.bboxes[idx]

        # Get specific frames and bounding boxes
        frames = os.listdir(os.path.join(self.videos_dir, video_id))
        frames = sorted(frames)
        act_frames = [frames[i] for i in indexes]

        # Load frames, crop and resize
        frames = []
        for i in range(len(act_frames)):
            frame = Image.open(os.path.join(self.videos_dir, video_id, act_frames[i]))
            w,h = frame.size
            if len(act_frames) != len(bboxes):
                print(len(act_frames), len(bboxes))
            abs_bbox = (int(bboxes[i][0] * w), int(bboxes[i][1] * h), int(bboxes[i][2] * w), int(bboxes[i][3] * h))
            frame = frame.crop((abs_bbox[0], abs_bbox[1], abs_bbox[2], abs_bbox[3]))
            frame = frame.resize((268, 198))
            #frame.save('frames/frame_{}.jpg'.format(i))
            frames.append(frame)
        #exit()

        # Pad or truncate frames
        if self.model_name == 'mvit':
            if len(frames) < 16:
                last_frame = frames[-1]
                frames.extend([last_frame] * (16 - len(frames)))
            elif len(frames) > 16:
                #frames = np.linspace(0, len(frames)-1, 16, dtype=int)
                indexes = np.linspace(0, len(frames)-1, 16, dtype=int)
                frames = [frames[i] for i in indexes]
        else:
            if len(frames) < 177:
                last_frame = frames[-1]
                frames.extend([last_frame] * (177 - len(frames)))
            elif len(frames) > 177:
                frames = frames[:177]
        
        # Adapt frames to model
        frames = torch.stack([torch.tensor(np.array(frame)) for frame in frames])
        frames = frames.permute(0, 3, 1, 2)
        preprocess = self.preprocess.transforms()
        frames = preprocess(frames)

        return frames, activity








