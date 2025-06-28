from torch.utils.data import Dataset

import os
import glob
import h5py
import torch

class VectorDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.embedding_files = []
        self.labels = []

        self.class_names = sorted([d.name for d in os.scandir(data_path) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        for class_name, label_idx in self.class_to_idx.items():
            class_path = os.path.join(data_path, class_name)
            for embedding_file in glob.glob(os.path.join(class_path, "*.hdf5")):
                self.embedding_files.append(embedding_file)
                self.labels.append(label_idx)

        print(
            f"Found {len(self.embedding_files)} embedding files across {len(self.class_names)} classes."
        )

    def __getitem__(self, index):
        if index < 0 or index >= len(self.embedding_files):
            raise IndexError("Index out of bounds for dataset.")
        file_path = self.embedding_files[index]
        label = self.labels[index]

        with h5py.File(file_path, "r") as f:
            # print(f"Keys in the HDF5 file: {list(f.keys())}")
            """
            ./embeddings/[train|validation|test]/<category_name>/
            │
            <video_id>.hdf5
            │
            ├─ raw_features/                                      # original, “low-level” metadata
            │  ├─ ffmpeg_numerical     float32[5]                 # [width, height, fps, sample_rate, channels]
            │  ├─ youtube_numerical    float32[5]                 # [view_count, like_count, comment_count, favorite_count, duration_sec]
            │  ├─ tags                 string[N_tags]             # variable-length list of tags
            │  ├─ rating               string                     # e.g. "ytAgeRestricted" or ""
            │  ├─ dimension            string                     # “2d” or “3d”
            │  ├─ definition           string                     # “hd” or “sd”
            │  ├─ projection           string                     # “rectangular” or “360”
            │  ├─ video_codec          string                     # e.g. "h264"
            │  ├─ audio_codec          string                     # e.g. "aac"
            │  ├─ published_at         string                     # ISO timestamp, e.g. "2025-05-27T03:14:15Z"
            │  ├─ channel_title        string                     # uploader’s channel name
            │  └─ default_language     string                     # e.g. "en"
            │
            └─ embedded_features/                                 # Third party embeddings and processed metadata
            ├─ audio_features                float32[1024]     # TwelveLabs audio embedding
            ├─ video_text_features           float32[1024]     # TwelveLabs visual-text embedding
            ├─ embedded_string_metadata      float32[3072]     # OpenAI embedding of string metadata
            └─ normalized_numerical_metadata float32[M_meta]   # L2-normalized numeric metadata vector
            """
            raw_features = f["raw_features"]
            visual_text_features = f["embedded_features/video_text_features"]
            audio_features = f["embedded_features/audio_features"]
            embedded_string_metadata = f["embedded_features/embedded_string_metadata"]
            # normalized_numerical_metadata = f[
            #     "embedded_features/normalized_numerical_metadata"
            # ]
            visual_text_features = torch.tensor(visual_text_features[:])
            audio_features = torch.tensor(audio_features[:])
            embedded_string_metadata = torch.tensor(embedded_string_metadata[:])

            visual_text_features = visual_text_features / visual_text_features.norm()
            audio_features = audio_features / audio_features.norm()
            embedded_string_metadata = (
                embedded_string_metadata / embedded_string_metadata.norm()
            )

        return visual_text_features, audio_features, embedded_string_metadata, label

    def __len__(self):
        return len(self.embedding_files)
