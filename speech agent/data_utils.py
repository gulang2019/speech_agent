"""Data preparation utilities for speech pipeline simulation."""

import os
from scipy.io import wavfile


def get_wav_info(file_path, window_size=0.025, hop_size=0.01):
    """
    Read WAV file and return its frame count and sample count.

    Args:
        file_path: Path to WAV file
        window_size: Window size in seconds (default: 0.025)
        hop_size: Hop size in seconds (default: 0.01)

    Returns:
        (frame_count, num_samples) or None if file not found
    """
    try:
        rate, data = wavfile.read(file_path)
        num_samples = data.shape[0]
        window_size_samples = int(window_size * rate)
        hop_size_samples = int(hop_size * rate)
        frame_count = (num_samples - window_size_samples) // hop_size_samples + 1
        return frame_count, num_samples
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return None, None


def prepare_data(dataset, dataset_path):
    """
    Prepare conversation list and user list from dataset.

    Each conversation alternates human/gpt turns. We extract:
    - voice metadata (frame counts, speech lengths)
    - user data structures with conversation history

    Args:
        dataset: HuggingFace dataset
        dataset_path: Root path for audio files

    Returns:
        userlist: list of UserData objects (only conversations with >1 turn)
    """
    # Import here to avoid circular dependency
    from data_models import VoiceMetadata, UserData

    convlist = []
    current_conv = []

    for i in range(len(dataset)):
        data = dataset[i]
        next_data = dataset[i + 1] if i + 1 < len(dataset) else None

        if next_data is None:
            break

        # Skip GPT-only entries, last human of a conversation, or invalid audio paths
        if data["from"] == "gpt" or \
           (data["from"] == "human" and next_data and next_data["conv_id"] != data["conv_id"]) or \
           ("valid_freq" not in data["audpath"]):
            continue

        # Validate human → gpt turn pattern
        assert data["from"] == "human", f"Expected 'human' but got {data['from']} at index {i}"
        assert next_data["from"] == "gpt", f"Expected 'gpt' but got {next_data['from']} at index {i + 1}"

        human_voice_path = os.path.join(dataset_path, data["file_name"])
        gpt_voice_path = os.path.join(dataset_path, next_data["file_name"])

        len_human_frames, _ = get_wav_info(human_voice_path)
        len_gpt_frames, _ = get_wav_info(gpt_voice_path)

        voice_metadata = VoiceMetadata(
            conv_id=data["conv_id"],
            len_human_speech=len(data["value"]),  # TODO: use actual tokenizer
            len_human_frames=len_human_frames,
            len_gpt_speech=len(next_data["value"]),  # TODO: use actual tokenizer
            len_gpt_frames=len_gpt_frames,
        )

        if current_conv and data["conv_id"] != current_conv[-1].conv_id:
            convlist.append(current_conv)
            current_conv = []

        current_conv.append(voice_metadata)

    # Add last conversation if exists
    if current_conv:
        convlist.append(current_conv)

    # Filter to conversations with more than 1 turn
    userlist = [UserData(convs=conv) for conv in convlist if len(conv) > 1]

    return userlist