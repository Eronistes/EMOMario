import os
import pandas as pd
import json

def load_participant_data(dataset_path):
    participants = []

    # Iterate over each participant folder
    participant_folders = sorted([f.path for f in os.scandir(dataset_path) if f.is_dir()])

    for participant_folder in participant_folders:
        participant_data = {}

        # Extract participant ID from folder name
        participant_id = os.path.basename(participant_folder)

        # Paths to sensor data
        sensor_folder = os.path.join(participant_folder, f"{participant_id}_sensor")
        acc_path = os.path.join(sensor_folder, "ACC.csv")
        bvp_path = os.path.join(sensor_folder, "BVP.csv")
        eda_path = os.path.join(sensor_folder, "EDA.csv")
        hr_path = os.path.join(sensor_folder, "HR.csv")
        ibi_path = os.path.join(sensor_folder, "IBI.csv")
        temp_path = os.path.join(sensor_folder, "TEMP.csv")

        # Load sensor data using pandas
        acc_data = pd.read_csv(acc_path) if os.path.exists(acc_path) else print('error acc')
        bvp_data = pd.read_csv(bvp_path) if os.path.exists(bvp_path) else print('error bvp')
        eda_data = pd.read_csv(eda_path) if os.path.exists(eda_path) else print('error eda')
        hr_data = pd.read_csv(hr_path) if os.path.exists(hr_path) else print('error hr')
        ibi_data = pd.read_csv(ibi_path) if os.path.exists(ibi_path) else print('error ibi')
        temp_data = pd.read_csv(temp_path) if os.path.exists(temp_path) else print('error temp')

        # Paths to JSON files
        gap_info_path = os.path.join(participant_folder, f"{participant_id}_gap_info.json")
        session_path = os.path.join(participant_folder, f"{participant_id}_session.json")
        video_info_path = os.path.join(participant_folder, f"{participant_id}_video_info.json")
        video_path = os.path.join(participant_folder, f"{participant_id}_video.avi")

        # Load JSON data
        with open(gap_info_path) as f:
            gap_info = json.load(f)

        with open(session_path) as f:
            session_info = json.load(f)

        with open(video_info_path) as f:
            video_info = json.load(f)

        # Store all data in participant_data dictionary
        participant_data['participant_id'] = participant_id
        participant_data['ACC'] = acc_data
        participant_data['BVP'] = bvp_data
        participant_data['EDA'] = eda_data
        participant_data['HR'] = hr_data
        participant_data['IBI'] = ibi_data
        participant_data['TEMP'] = temp_data
        participant_data['gap_info'] = gap_info
        participant_data['session_info'] = session_info
        participant_data['video_info'] = video_info
        participant_data['video_path'] = video_path

        participants.append(participant_data)

    return participants

def load_single_participant(dataset_path, participant_number):
    # Get a sorted list of participant folders
    participant_folders = sorted([f.path for f in os.scandir(dataset_path) if f.is_dir()])

    # Check if the participant number is valid
    if participant_number < 0 or participant_number >= len(participant_folders):
        raise ValueError(f"Invalid participant number: {participant_number}. Must be between 0 and {len(participant_folders) - 1}.")

    # Get the folder for the specified participant
    participant_folder = participant_folders[participant_number]
    participant_data = {}

    # Extract participant ID from folder name
    participant_id = os.path.basename(participant_folder)

    # Paths to sensor data
    sensor_folder = os.path.join(participant_folder, f"{participant_id}_sensor")
    acc_path = os.path.join(sensor_folder, "ACC.csv")
    bvp_path = os.path.join(sensor_folder, "BVP.csv")
    eda_path = os.path.join(sensor_folder, "EDA.csv")
    hr_path = os.path.join(sensor_folder, "HR.csv")
    ibi_path = os.path.join(sensor_folder, "IBI.csv")
    temp_path = os.path.join(sensor_folder, "TEMP.csv")

    # Load sensor data using pandas
    acc_data = pd.read_csv(acc_path) if os.path.exists(acc_path) else print(f'Error: ACC data not found for participant {participant_id}')
    bvp_data = pd.read_csv(bvp_path) if os.path.exists(bvp_path) else print(f'Error: BVP data not found for participant {participant_id}')
    eda_data = pd.read_csv(eda_path) if os.path.exists(eda_path) else print(f'Error: EDA data not found for participant {participant_id}')
    hr_data = pd.read_csv(hr_path) if os.path.exists(hr_path) else print(f'Error: HR data not found for participant {participant_id}')
    ibi_data = pd.read_csv(ibi_path) if os.path.exists(ibi_path) else print(f'Error: IBI data not found for participant {participant_id}')
    temp_data = pd.read_csv(temp_path) if os.path.exists(temp_path) else print(f'Error: TEMP data not found for participant {participant_id}')

    # Paths to JSON files
    gap_info_path = os.path.join(participant_folder, f"{participant_id}_gap_info.json")
    session_path = os.path.join(participant_folder, f"{participant_id}_session.json")
    video_info_path = os.path.join(participant_folder, f"{participant_id}_video_info.json")
    video_path = os.path.join(participant_folder, f"{participant_id}_video.avi")

    # Load JSON data
    with open(gap_info_path) as f:
        gap_info = json.load(f)

    with open(session_path) as f:
        session_info = json.load(f)

    with open(video_info_path) as f:
        video_info = json.load(f)

    # Store all data in participant_data dictionary
    participant_data['participant_id'] = participant_id
    participant_data['ACC'] = acc_data
    participant_data['BVP'] = bvp_data
    participant_data['EDA'] = eda_data
    participant_data['HR'] = hr_data
    participant_data['IBI'] = ibi_data
    participant_data['TEMP'] = temp_data
    participant_data['gap_info'] = gap_info
    participant_data['session_info'] = session_info
    participant_data['video_info'] = video_info
    participant_data['video_path'] = video_path

    return participant_data


# Example usage:
dataset_path = "toadstool/participants"
participants_data = load_participant_data(dataset_path)
"""
# Now participants_data is a list of dictionaries, each containing data for one participant
# You can access individual participant's data like this:
for participant in participants_data:
    print(f"Participant ID: {participant['participant_id']}")
    print(f"Number of ACC samples: {len(participant['ACC']) if participant['ACC'] is not None else 'N/A'}")
    print(f"Number of BVP samples: {len(participant['BVP']) if participant['BVP'] is not None else 'N/A'}")
    print(f"Session start time: {participant['session_info']['score']}")
    print(f"Number of gaps: {len(participant['gap_info']['indices'])}")
    print(f"Video duration: {participant['video_info']['start_time']} seconds")
    print(f"---------------------------------------")"""