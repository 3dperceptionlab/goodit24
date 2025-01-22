"""
- Reads the annotations and extracts the statistics of the activities.

Usage:
    extract_statistics.py <input_file> <output_file>

Options:
    -h --help       Show this screen.
    <input_file>      Input file path with annotations.
    <output_file>    Output file path.
"""

import csv
from collections import defaultdict
import statistics
import math
from queue import Queue
import heapq
from docopt import docopt


def parse_annotations(file_path):
    """Parse the annotations from the input file and return a dictionary with the data."""
    data = defaultdict(lambda: defaultdict(list))
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_path, frame_id, xtl, ytl, xbr, ybr, activity_id, class_id = row
            video_id = image_path.split('/')[-2]
            frame_id = int(frame_id)
            activity_id = int(activity_id)
            class_id = int(class_id)
            bbox = (float(xtl), float(ytl), float(xbr), float(ybr))
            data[video_id][class_id].append((frame_id, activity_id, bbox, image_path))
    return data


def calculate_distance(box1, box2):
    """Calculate the Euclidean distance between the centers of two bounding boxes. **Working better than the second one"""
    center_x1 = (box1[0] + box1[2]) / 2
    center_y1 = (box1[1] + box1[3]) / 2
    center_x2 = (box2[0] + box2[2]) / 2
    center_y2 = (box2[1] + box2[3]) / 2
    return math.sqrt((center_x1 - center_x2) * 2 + (center_y1 - center_y2) * 2)

# def calculate_distance(box1, box2):
#     """Calculate the sum of absolute differences between corresponding coordinates of two bounding boxes."""
#     # Unpack the bounding box coordinates
#     xtl1, ytl1, xbr1, ybr1 = box1
#     xtl2, ytl2, xbr2, ybr2 = box2

#     # Calculate absolute differences for each coordinate and sum them
#     distance = (abs(xtl1 - xtl2) + abs(ytl1 - ytl2) +
#                 abs(xbr1 - xbr2) + abs(ybr1 - ybr2))
#     return distance


def checkEqualSubjects(actual_subjects_range1, actual_subjects_range2, activity_durations, is_last_frame=False):
    """Check for equal subjects in both ranges and update the activity durations."""
    # Priority queue to store potential matches based on the distance
    potential_matches = []

    # Create a list of potential matches with distances
    for sub_id1, (_, frame_id1, activity_id1, bbox1) in actual_subjects_range1.items():
        for sub_id2, (_, frame_id2, activity_id2, bbox2) in actual_subjects_range2.items():
            distance = calculate_distance(bbox1, bbox2)
            heapq.heappush(potential_matches, (distance, sub_id1, sub_id2))

    # To keep track of matched subjects in range2
    matched_subjects_range2 = set()
    best_matches = {}

    # Process the priority queue to find the best match for each subject in range1
    while potential_matches:
        distance, sub_id1, sub_id2 = heapq.heappop(potential_matches)
        if sub_id1 not in best_matches and sub_id2 not in matched_subjects_range2:
            best_matches[sub_id1] = sub_id2
            matched_subjects_range2.add(sub_id2)

    aux_actual_subjects_range1 = {} # Auxiliary dictionary to store the updated subjects in range 1, so that posterior access is not affected
    # Update activity durations based on matched subjects
    for sub_id1, sub_id2 in best_matches.items():
        start_frame_id1, frame_id1, activity_id1, bbox = actual_subjects_range1[sub_id1]
        _, frame_id2, activity_id2, bbox = actual_subjects_range2[sub_id2]

        actual_subjects_range1.pop(sub_id1)  # Remove the previous subject from range1

        if activity_id1 != activity_id2: # Activity change
            if frame_id2 - start_frame_id1 > 128: # Check if the activity duration is more than 2 frames
                activity_durations[activity_id1].append(frame_id2 - start_frame_id1)  # End the current duration
            aux_actual_subjects_range1[sub_id2] = (frame_id2, frame_id2, activity_id2, bbox)  # Update the subjects in range 1
            if is_last_frame: # Last frame
                if frame_id2 - frame_id2 + 1 > 128: # Check if the activity duration is more than 2 frames
                    activity_durations[activity_id2].append(frame_id2 - frame_id2 + 1)
        else: # Same activity
            aux_actual_subjects_range1[sub_id2] = (start_frame_id1, frame_id2, activity_id1, bbox)  # Update the subjects in range 1
            if is_last_frame: # Last frame
                if frame_id2 - start_frame_id1 + 1 > 128: # Check if the activity duration is more than 2 frames
                    activity_durations[activity_id1].append(frame_id2 - start_frame_id1 + 1)

    # Check for unpaired subjects in both ranges and terminate their activities, meaning that either a subject left the frame or a new subject appeared
    unpaired_subjects_range1 = set(actual_subjects_range1.keys()) - set(best_matches.keys())
    unpaired_subjects_range2 = set(actual_subjects_range2.keys()) - matched_subjects_range2

    # Terminate the activities of subjects which are not anymore in the frame
    for sub_id1 in unpaired_subjects_range1:
        start_frame_id1, frame_id1, activity_id1, _ = actual_subjects_range1[sub_id1]
        actual_subjects_range1.pop(sub_id1)  # Remove the previous subject from range1
        if frame_id1 - start_frame_id1 > 128: # Check if the activity duration is more than 2 frames
            activity_durations[activity_id1].append(frame_id1 - start_frame_id1)  # Terminate the activity of subjects which are not anymore in the frame

    # New subjects appeared
    for sub_id2 in unpaired_subjects_range2:
        start_frame_id2, frame_id2, activity_id2, bbox = actual_subjects_range2[sub_id2]
        actual_subjects_range1[sub_id2] = (frame_id2, frame_id2, activity_id2, bbox)  # New subject appeared
        if is_last_frame: # Last frame
            if frame_id2 - frame_id2 + 1 > 128: # Check if the activity duration is more than 2 frames
                activity_durations[activity_id2].append(frame_id2 - frame_id2 + 1)

    actual_subjects_range1.update(aux_actual_subjects_range1) # Update the subjects in range 1 with the auxiliary dictionary, range 1 is now the previous range 2


def compute_durations(data):
    """Compute the durations of the activities in the given data by iterating over it, saving 2 windows for checking the frames with previous frames and knowing which one belongs to each."""
    activity_durations = defaultdict(list)
    for video_id, class_data in data.items():
        print(f"Processing video ID: {video_id}")
        for class_id, frames in class_data.items(): # Iterate over the videos
            frames.sort()  # Sort frames by frame_id
            previous_frame_id = None
            different_frames_count = 0 # Count of different frames, used for knowing when to check the subjects (meaning that the 2 windows are full)
            subject_id_range1, subject_id_range2 = 0, 0  # Subject IDs for the 2 windows
            actual_subjects_range1, actual_subjects_range2 = {}, {} # Subjects for the 2 windows

            for i in range(len(frames)): # Iterate over the frames of the video
                frame_id, activity_id, bbox, _ = frames[i] # Get the frame data

                if previous_frame_id is None: # First frame
                    previous_frame_id = frame_id

                if i == 0: # First frame
                    start_frame_id = frame_id
                
                if frame_id != previous_frame_id: # Different frame
                    different_frames_count += 1
                    if different_frames_count > 1: # 2nd different frame, check the subjects (2 windows of frames)
                        different_frames_count = 1

                        checkEqualSubjects(actual_subjects_range1, actual_subjects_range2, activity_durations)

                        # Assign the current frame to the second range, as the first range will be the previous 2nd range
                        actual_subjects_range2 = {}
                        subject_id_range2 = 0
                        actual_subjects_range2[subject_id_range2] = (start_frame_id, frame_id, activity_id, bbox)
                        subject_id_range2 += 1
                    else: # 1st different frame number, assign the current frame to the second range
                        actual_subjects_range2[subject_id_range2] = (start_frame_id, frame_id, activity_id, bbox)
                        subject_id_range2 += 1
                else: # Same frame
                    if different_frames_count == 0: # Same frame as the previous one
                        actual_subjects_range1[subject_id_range1] = (start_frame_id, frame_id, activity_id, bbox)
                        subject_id_range1 += 1
                    else: # Second window of frames
                        actual_subjects_range2[subject_id_range2] = (start_frame_id, frame_id, activity_id, bbox)
                        subject_id_range2 += 1

                previous_frame_id = frame_id # Update the previous frame ID

            # Process the last frame or frames
            checkEqualSubjects(actual_subjects_range1, actual_subjects_range2, activity_durations, is_last_frame=True)

    return activity_durations

def compute_statistics(durations):
    stats = {}
    all_durations = []
    for activity_id, duration_list in durations.items():
        all_durations.extend(duration_list)
        mean_duration = statistics.mean(duration_list)
        median_duration = statistics.median(duration_list)
        max_duration = max(duration_list)
        min_duration = min(duration_list)
        stats[activity_id] = {
            'mean': mean_duration,
            'median': median_duration,
            'max': max_duration,
            'min': min_duration,
            'countOfSegments': len(duration_list),
            'countOfActivityFrames': sum(duration_list)
        }

    # Sort stats by activity ID
    stats = dict(sorted(stats.items()))

    overall_stats = {
        'mean': statistics.mean(all_durations) if all_durations else 0,
        'median': statistics.median(all_durations) if all_durations else 0,
        'max': max(all_durations) if all_durations else 0,
        'min': min(all_durations) if all_durations else 0,
        'countOfSegments': len(all_durations),
        'countOfActivityFrames': sum(all_durations)
    }

    return stats, overall_stats

def main(file_path, output_file):
    """Main"""
    data = parse_annotations(file_path) # Parse the annotations from the input file
    durations = compute_durations(data) # Compute the durations of the activities
    stats, overall_stats = compute_statistics(durations) # Compute the statistics of the activities
    
    # Write the statistics to the output file
    with open(output_file, 'w') as file:
        file.write("--Overall Statistics--\n")
        for key, value in overall_stats.items():
            file.write(f"{key.capitalize()}: {value}\n")
    
        file.write("\n--Statistics by Activity--\n")
        for activity_id, stat in stats.items():
            file.write(f"\nActivity ID {activity_id}:\n")
            for key, value in stat.items():
                file.write(f"  {key.capitalize()}: {value}\n")

if _name_ == "_main_":
    arguments = docopt(_doc_)
    file_path = arguments['<input_file>']
    output_file = arguments['<output_file>']
    main(file_path, output_file)


# Videos raising problems:

# 024. birds too close in 365-366, 1934, 2131

# 047. problems for many birds

# 049. problems for many birds

# **072. problems with too close birds, just raising with calculate_distances function(2), the one made checking coordinate by coordinate instead of centroid

# 076. problems with adapt_activities

# 140. strange activity problem