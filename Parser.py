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
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# --------------------------------------------------------------------------- #
#	 CONFIGURATION                                                              #
# --------------------------------------------------------------------------- #
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
youtube = build('youtube', 'v3', developerKey='')
# --------------------------------------------------------------------------- #
#  CONSTANTS                                                                  #
# --------------------------------------------------------------------------- #
YT_API_BATCH: int = 50
DESIRED_CATEGORIES: int = 10
CROSS_SPLIT_SHARDS_IDS: set[str] = set()
VOCABULARY_PATH: str = './data/vocabulary.csv'
DESIRED_VIDEO_PER_CATEGORY: dict[str, int] = {
		'train': 144,
		'validation': 18,
		'test': 18
}
# --------------------------------------------------------------------------- #
#  REGEX PATTERNS                                                             #
# --------------------------------------------------------------------------- #
_KEY_RE          = re.compile(r'^[A-Za-z0-9_-]{4}$')
_NEW_WRAPPER_RE  = re.compile(r'\((\{.*\"vid\".*})\)')
_OLD_WRAPPER_RE  = re.compile(r'i\(\s*"[^"]+"\s*,\s*"([^"]{11})"\s*\)')
# --------------------------------------------------------------------------- #
#  FEATURE SPECS SCHEMA (YouTube-8M video-level)                              #
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

def build_categories_from_csv_dict(external_call: bool = False) -> dict[str, set[int]] or None:
	"""
	Builds a dictionary of macro categories and their respective subcategories IDs from a CSV file.

	:return: dictionary of macro categories and their subcategories IDs
	"""
	with open(VOCABULARY_PATH, 'r', encoding='utf-8') as file:
		reader = csv.DictReader(file)
		for row in reader:
			if row['Vertical1'] != '(Unknown)':
				if row['Vertical1'] not in m_categories:
					m_categories[row['Vertical1']] = set()
				m_categories[row['Vertical1']].add(int(row['Index']))
		if external_call:
			return m_categories.copy()
		return None

def iterate_shards(root_path: typing.Union[str, pathlib.Path],
                  split: str, pattern: str = '*.tfrecord') -> Generator[Any, Any, None]:
	"""
	Iterates over TFRecord shards in a directory.

	:param root_path: path to the directory containing TFRecord shards
	:param split: data split (train, validation, test)
	:param pattern: file pattern to match (default: '*.tfrecord')
	:return: generator yielding parsed TFRecord examples
	"""
	shard_counter: int = 0
	root = pathlib.Path(root_path)
	for shard_path in root.glob(pattern):
		shard_counter += 1
		print(f'[{time.strftime("%H:%M:%S")} | {split}] Processing shard: {shard_path} #{shard_counter}')
		if split == 'validation':
			CROSS_SPLIT_SHARDS_IDS.add(shard_path.stem)
		if split == 'test' and shard_path.stem in CROSS_SPLIT_SHARDS_IDS:
			continue
		dataset = tf.data.TFRecordDataset(str(shard_path), compression_type='')
		for record in dataset:
			yield tf.io.parse_single_example(record, features)

def find_macro_category(vid_labels: set) -> Optional[str]:
	"""
	Finds the category name corresponding to a set of video IDs.

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

def are_videos_available(yt_ids: list[str]) -> set[str]:
	"""
	Checks if a batch of YouTube videos are available.

	:param yt_ids: YouTube video IDs (11-character string)
	:return: set of available YouTube video IDs that are public and embeddable
	"""
	try:
		request = youtube.videos().list(part='status,contentDetails', id=','.join(yt_ids))
		response = request.execute()
	except HttpError as e:
		print(f'[{time.strftime("%H:%M:%S")}] Error fetching video details: {e}')
		return set()

	available_videos:	set[str] = set()
	for item in response.get('items', []):
		status = item.get('status', {})
		content_details = item.get('contentDetails', {})
		if status.get('privacyStatus') !=	'public':
			continue
		if not status.get('embeddable', True):
			continue
		if content_details.get('contentRating', {}).get('ytRating') == 'ytAgeRestricted':
			continue
		region = content_details.get('regionRestriction', {})
		if 'blocked' in region or 'blockedIn' in region or 'allowed' in region:
			continue
		available_videos.add(item['id'])
	return available_videos

def yt8m_key_to_video(key: str, timeout: int = 4) -> Optional[tuple[str, str]]:
	"""
	Converts a YouTube-8M video key to a YouTube URL.

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
			yt_id = match.group(1)
		else:
			match = _NEW_WRAPPER_RE.search(payload)
			if not match:
				return None
			yt_id = json.loads(match.group(1)).get('vid')
		if yt_id and len(yt_id) == 11:
			return f'https://www.youtube.com/watch?v={yt_id}', yt_id
		return None
	except requests.RequestException:
		return None

def is_split_full(video_urls: dict[str, dict[str, str]], quota:	int, needed_categories: set[str]) -> bool:
	"""
	Checks if the video URLs dictionary is full for the given quota and categories.

	:param video_urls: dictionary of video URLs for each category
	:param quota: number of videos needed per category
	:param needed_categories: set of categories to check
	:return: True if the dictionary is full, otherwise False
	"""
	return (len(needed_categories) == DESIRED_CATEGORIES and
	        all(len(video_urls.get(category, ())) == quota for category in needed_categories))

def export_urls_to_tsv(split: str, urls: dict[str, dict[str, str]]) -> None:
	"""
	Exports video URLs to CSV files for each category.

	:param split: data split (train, validation, test)
	:param urls: dictionary of video URLs for each category
	"""
	tsv_path = pathlib.Path(f'./data/{split}/urls/')
	tsv_path.mkdir(parents=True, exist_ok=True)
	for m_category, url_map in urls.items():
		with open(tsv_path / f'{m_category.replace(" ", "_")}.tsv', 'w', encoding='utf-8') as file:
			file.write('video_url\tvideo_id\n')
			for yt_id, url in url_map.items():
				file.write(f'{url}\t{yt_id}\n')
		print(f'[{time.strftime("%H:%M:%S")} | {split}] '
		      f'Exported {len(url_map)} URLs for category "{m_category}" to {tsv_path / f"{m_category}.tsv"}')

def init_dataset() -> None:
	"""
	Samples YouTube URLs from the shard files for each category and split, according to the given ratios.
	"""
	for split in ['train', 'validation', 'test']:
		initialized: bool = False
		pending: list[tuple[str, str]] = []
		video_urls: dict[str, dict[str, str]] = {}
		quota: int = DESIRED_VIDEO_PER_CATEGORY[split]
		wanted_categories: set[str] = sel_m_categories.copy()

		def flush_batch() -> None:
			nonlocal pending
			sanitized_pending = are_videos_available([video_id for (video_id, _) in pending])
			for video, category in pending:
				if video in sanitized_pending and len(video_urls.get(category, ())) < quota:
					video_urls.setdefault(category, {})[video] = f'https://www.youtube.com/watch?v={video}'
			pending: list[tuple[str, str]] = []

		for shard in iterate_shards(f'./data/{split}', split):
			labels = set(shard['labels'].values.numpy())
			cat = find_macro_category(labels)
			if cat is None:
				continue
			if split == 'train':
				if cat not in wanted_categories:
					if len(wanted_categories) >= DESIRED_CATEGORIES:
						continue
					wanted_categories.add(cat)
			elif cat not in wanted_categories:
				continue
			if len(video_urls.get(cat, ())) >=	quota:
				continue
			video_info = yt8m_key_to_video(shard['id'].numpy().decode('utf-8'))
			if video_info is not None:
				pending.append((video_info[1], cat))
				if len(pending) >= YT_API_BATCH:
					flush_batch()
			if is_split_full(video_urls, quota, wanted_categories):
				initialized = True
				export_urls_to_tsv(split, video_urls)
				break
			if not initialized:
				print(f'[{time.strftime("%H:%M:%S")} | {split}] Failed to collect {quota} videos for each category')
				print(f'[{time.strftime("%H:%M:%S")} | {split}] Please adjust the split size or download more shards')
				exit(1)
			if split == 'train':
				sel_m_categories.update(wanted_categories)

if __name__ == '__main__':
		build_categories_from_csv_dict()
		init_dataset()
		exit(0)
