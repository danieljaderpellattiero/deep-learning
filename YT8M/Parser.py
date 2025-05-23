import re
import csv
import time
import json
import typing
import pathlib
import urllib3
import requests
import tensorflow as tf
from typing import Any, Generator, Optional
# --------------------------------------------------------------------------- #
#		CONFIGURATION                                                              #
# --------------------------------------------------------------------------- #
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# --------------------------------------------------------------------------- #
#  CONSTANTS                                                                  #
# --------------------------------------------------------------------------- #
DESIRED_CATEGORIES: int = 10
CROSS_SPLIT_SHARDS_IDS: set[str] = set()
SPLITS: list = ['train', 'validation', 'test']
VOCABULARY_PATH: str = './data/vocabulary.csv'
DESIRED_VIDEO_PER_CATEGORY: dict[str, int] = {
		'train': 144,
		'validation': 18,
		'test': 18,
}
# --------------------------------------------------------------------------- #
#  REGEX PATTERNS                                                             #
# --------------------------------------------------------------------------- #
_KEY_RE          = re.compile(r'^[A-Za-z0-9_-]{4}$')
_NEW_WRAPPER_RE  = re.compile(r'\((\{.*\"vid\".*})\)')
_OLD_WRAPPER_RE  = re.compile(r'i\(\s*"[^"]+"\s*,\s*"([^"]{11})"\s*\)')
# --------------------------------------------------------------------------- #
#  FEATURE SPEC (YouTube-8M video-level)                                      #
# --------------------------------------------------------------------------- #
features: dict = {
		'id': tf.io.FixedLenFeature([], tf.string),
		'labels': tf.io.VarLenFeature(tf.int64),
}
# --------------------------------------------------------------------------- #
#  VIDEO MACRO CATEGORIES AND RESPECTIVE SUBCATEGORIES IDS                    #
# --------------------------------------------------------------------------- #
sel_m_categories = set()
m_categories:	dict[str, set[int]] = {}

def build_categories_from_csv_dict() -> None:
		"""
		Build a dictionary of macro categories and their respective subcategories IDs from a CSV file.
		:return: dictionary of macro categories and their subcategories IDs
		"""
		with open(VOCABULARY_PATH, 'r', encoding='utf-8') as file:
				reader = csv.DictReader(file)
				for row in reader:
						if row['Vertical1'] != '(Unknown)':
								if row['Vertical1'] not in m_categories:
										m_categories[row['Vertical1']] = set()
								m_categories[row['Vertical1']].add(int(row['Index']))

def iterate_shard(root_dir: typing.Union[str, pathlib.Path],
                  split: str, pattern: str = '*.tfrecord') -> Generator[Any, Any, None]:
		"""
		Iterate over TFRecord shards in a directory.
		:param root_dir: path to the directory containing TFRecord shards
		:param split: data split (train, validation, test)
		:param pattern: file pattern to match (default: '*.tfrecord')
		:return: generator yielding parsed TFRecord examples
		"""
		shard_counter: int = 0
		root = pathlib.Path(root_dir)
		for shard_path in root.glob(pattern):
				shard_counter += 1
				print(f'Processing shard: {shard_path} \t #{shard_counter}')
				if split == 'validation':
						CROSS_SPLIT_SHARDS_IDS.add(shard_path.stem)
				if split == 'test' and shard_path.stem in CROSS_SPLIT_SHARDS_IDS:
						continue
				dataset = tf.data.TFRecordDataset(str(shard_path), compression_type='')
				for record in dataset:
						yield tf.io.parse_single_example(record, features)

def find_macro_category(vid_labels: set) -> Optional[str]:
		"""
		Find the category name corresponding to a set of video IDs.
		:param vid_labels: set of video category labels
		:return: name of the macro category if found, otherwise None
		"""
		found = None
		for (name, ids) in m_categories.items():
				if vid_labels.issubset(ids):
						if found is not None:
								return None
						found = name
		return found

def yt8m_key_to_url(key: str, timeout: int = 4) -> Optional[str]:
		"""
		Convert a YouTube-8M video key to a YouTube URL.
		:param key: YouTube-8M video key (4-character string)
		:param timeout: timeout for the HTTP request (default: 4 seconds)
		:return: YouTube URL if found, otherwise None
		"""
		if not _KEY_RE.fullmatch(key):
			return None
		yt_lookup_url = f'https://data.yt8m.org/2/j/i/{key[:2]}/{key}.js'
		try:
				response = requests.get(yt_lookup_url, timeout=timeout, verify=False)
				if response.status_code != 200:
						return None
				payload = response.text
				match = _OLD_WRAPPER_RE.search(payload)
				if match:
					return f'https://www.youtube.com/watch?v={match.group(1)}'
				match = _NEW_WRAPPER_RE.search(payload)
				if match:
					vid = json.loads(match.group(1)).get('vid')
					if vid and len(vid) == 11:
						return f'https://www.youtube.com/watch?v={vid}'
		except requests.RequestException	as e:
				print(f'Error fetching URL: {e}')
		return None

def is_split_full(video_urls: dict[str, set[str]], quota:	int, needed_categories: set[str]) -> bool:
		"""
		Check if the video URLs dictionary is full for the given quota and categories.
		:param video_urls: dictionary of video URLs for each category
		:param quota: number of videos needed per category
		:param needed_categories: set of categories to check
		:return: True if the dictionary is full, otherwise False
		"""
		return (len(needed_categories) == DESIRED_CATEGORIES and
		        all(len(video_urls.get(category, ())) == quota for category in needed_categories))

def export_urls_to_csv(split: str, urls: dict[str, set[str]]) -> None:
		"""
		Export video URLs to CSV files for each category.
		:param split: data split (train, validation, test)
		:param urls: dictionary of video URLs for each category
		"""
		csv_path = pathlib.Path(f'./data/{split}/urls/')
		csv_path.mkdir(parents=True, exist_ok=True)
		for m_category, url_set in urls.items():
				with open(csv_path / f'{m_category.replace(" ", "_")}.csv', 'w', newline='', encoding='utf-8') as file:
						writer = csv.writer(file)
						for url in url_set:
								writer.writerow([url])
				print(f'Exported {len(url_set)} URLs for category "{m_category}" to {csv_path / f"{m_category}.csv"}')

def init_splits() -> None:
		"""
		Sample YouTube URLs from the shard files for each category and split, according to the given ratios.
		"""
		for split in SPLITS:
				initialized: bool = False
				quota: int = DESIRED_VIDEO_PER_CATEGORY[split]
				video_urls: dict[str, set[str]] = {}
				wanted_categories: set[str] = sel_m_categories.copy()
				for shard in iterate_shard(f'./data/{split}', split):
						# Classify the video labels
						labels = set(shard['labels'].values.numpy())
						cat = find_macro_category(labels)
						if cat is None:
								continue
						# Check if the category is allowed in the current split
						if split == 'train':
								if cat not in wanted_categories:
										if len(wanted_categories) >= DESIRED_CATEGORIES:
												continue
										else:
												wanted_categories.add(cat)
						elif cat not in wanted_categories:
								continue
						# Add the video URL to the category bucket if it exists
						bucket = video_urls.setdefault(cat, set())
						if len(bucket) >= quota:
								continue
						video_url = yt8m_key_to_url(shard['id'].numpy().decode('utf-8'))
						if video_url is not None:
								bucket.add(video_url)
						# ---
						if is_split_full(video_urls, quota, wanted_categories):
								initialized = True
								export_urls_to_csv(split, video_urls)
								break
				if not initialized:
						print(f'Failed to collect {quota} videos for each category in {split} split.')
						print('Please adjust the split size or download more shards.')
						exit(1)
				if split == 'train':
						sel_m_categories.update(wanted_categories)

if __name__ == "__main__":
		t0: time =	time.time()
		build_categories_from_csv_dict()
		init_splits()
		t1: time =	time.time()
		print(f'Execution time: {t1 - t0:.2f} seconds')
		exit(0)
