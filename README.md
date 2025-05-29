# YT8M

> Utility for YouTube-8M dataset

## Video-level features dataset (May 14th, 2018 release)

> 6.1M videos, 3862 classes, 3.0 labels/video, 2.6B audio-visual features

Video-level features are stored as tensorflow.Example protocol buffers.  
A tensorflow.Example proto is reproduced here in text format:

```python
features: {
  feature: {
    key  : "id"
    value: {
      bytes_list: {
        value: (Video id)
      }
    }
  }
  feature: {
    key  : "labels"
    value: {
      int64_list: {
        value: [1, 522, 11, 172]  # label list
      }
    }
  }
  feature: {
    # Average of all 'rgb' features for the video
    key  : "mean_rgb"
    value: {
      float_list: {
        value: [1024 float features]
      }
    }
  }
  feature: {
    # Average of all 'audio' features for the video
    key  : "mean_audio"
    value: {
      float_list: {
        value: [128 float features]
      }
    }
  }
}
```

The total size of the video-level features is 31 Gigabytes.  
They are broken into 3844 shards which can be subsampled to reduce the dataset size.

## Data preprocessing pipeline

### Phase 1 － Download the dataset (shards)

```bash
# Preliminary steps
mkdir -p ./data
mkdir -p ./data/train
mkdir -p ./data/validation
mkdir -p ./data/test
# Download the vocabulary csv file inside ./data
wget https://research.google.com/youtube8m/csv/2/vocabulary.csv
# Retrieve the dataset downloader script inside ./data
wget data.yt8m.org/download.py
# Download the video-level features
# The shards have to be stored in their own directory
# Validation shards are also used for the test set, via cross-splitting
cat ../download.py | shard=1,10 partition=2/video/train mirror=asia python
cat ../download.py | shard=1,10 partition=2/video/validate mirror=asia python
```

### Phase 2 － YouTube videos parsing

Run the `Parser.py` module after setting up the virtual environment and installing the dependencies.  

```bash
python Parser.py
```

**How it works:**

1. Parse YouTube video anonymous ids from the downloaded dataset shards by macro category.
2. The real urls will be then derived from the ids and stored in a _.tsv_ file.

> Note: The ratios of the splits are 80% train, 10% validation and 10% test and are hardcoded in the script.

### Phase 3 － YouTube videos download and clipping

Run the `Downloader.py` module after setting up the virtual environment and installing the dependencies.  

```bash
python Downloader.py
```

**How it works:**

1. Downloads the YouTube videos using the URLs stored in the _.tsv_ files.
2. The videos will be clipped to the desired duration and stored in the appropriate directory.

> Note: The default duration is 20 seconds and its hardcoded in the script.

### Phase 4 － Video features extraction

Run the `Featurizer.py` module after setting up the virtual environment and installing the dependencies.  

```bash
python Featurizer.py
```

**How it works:**

1. Retrieves technical features using the `ffmpeg` tool.
2. Retrieves analytic features using the YouTube API v3.
3. Normalizes numerical features and embeds categorical features with OpenAI API.
4. Retrieves the video-text and audio features embeddings with TwelveLabs API. (Marengo 2.7)
5. Saves the features in a _.hdf5_ file.

### BobTheDownloader

If a _.tsv_ file contains YouTube urls that somehow cannot be downloaded, you can run the `BobTheDownloader.py` module to regenerate the file with fresh urls retrieved from the shards. _"Can we fix it? Yes we can!"_

### BobTheClipper

If a video doesn't not match TwelveLabs' requirements, you can run the `BobTheClipper.py` module to clip it to the closest aspect ratio supported, by adding black bars to the sides or top/bottom of the video.

## HDF5 File structure

The HDF5 file structure follows:

```python
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
```

> To do:
>
> - Blacklists for broken urls and used shards. (`Downloader.py` -> `Parser.py` -> `Bob.py`)
> - Validation and test sets shard tracking. (for _.tsv_ file generation)
> - Clip videos after downloading or check before downloading.
