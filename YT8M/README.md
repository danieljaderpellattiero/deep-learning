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

### Phase 2 － YouTube videos sampling

Run the `Sampler.py` module after setting up the virtual environment and installing the dependencies.  

```bash
python Sampler.py
```

**How it works:**

1. The script will sample YouTube video anonymous ids from the downloaded dataset shards by macro category.
2. The real urls will be then derived from the ids and stored in a csv file.

> Note: The ratios of the splits are 80% train, 10% validation and 10% test and are hardcoded in the script.
