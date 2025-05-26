import re
import csv
import time
import json
import typing
import urllib3
import pathlib
import requests
import tensorflow as tf
from Downloader import download_and_clip
from typing import Generator, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
youtube = build('youtube', 'v3', developerKey='AIzaSyBQ8JQ54N1rY6wBXqd8ftZAXcL55zIGDpY')

_KEY_RE          = re.compile(r'^[A-Za-z0-9_-]{4}$')
_NEW_WRAPPER_RE  = re.compile(r'\((\{.*\"vid\".*})\)')
_OLD_WRAPPER_RE  = re.compile(r'i\(\s*"[^"]+"\s*,\s*"([^"]{11})"\s*\)')

YT_API_BATCH: int = 50
DATA_PATH:	str = '../data'
VIDEO_PATH: str = '../videos'
VOCABULARY_PATH: str = '../data/vocabulary.csv'
DESIRED_VIDEO_PER_CATEGORY: dict[str, int] = {
		'train': 144,
		'validation': 18,
		'test': 18
}

features_schema: dict = {
		'id': tf.io.FixedLenFeature([], tf.string),
		'labels': tf.io.VarLenFeature(tf.int64),
}

sel_m_categories = set()
m_categories: dict[str, set[int]] = {}

def build_categories_from_csv_dict() -> None:
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

def iterate_shards(root_dir: typing.Union[str, pathlib.Path],
                   split: str, pattern: str = "*.tfrecord") -> Generator[dict[str, Any], None, None]:
		shard_counter: int = 0
		root = pathlib.Path(root_dir)
		for shard_path in root.glob(pattern):
				shard_counter += 1
				print(f'[{time.strftime("%H:%M:%S")} | {split}] Processing shard: {shard_path} #{shard_counter}')
				dataset = tf.data.TFRecordDataset(str(shard_path), compression_type='')
				for record in dataset:
						yield tf.io.parse_single_example(record, features_schema)

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

def export_urls_to_tsv(split: str, category: str, video_ids: set[str]) -> None:
		"""
		Exports video URLs to a TSV file.

		:param split: dataset split (train, validation, test)
		:param category: category of the videos
		:param video_ids: set of video IDs to export
		"""
		with open(f'{DATA_PATH}/{split}/urls/{category}_reloaded.tsv', 'w', encoding='utf-8') as file:
				file.write('video_url\tvideo_id\n')
				for video_id in sorted(video_ids):
						file.write(f'https://www.youtube.com/watch?v={video_id}\t{video_id}\n')

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

		available_videos: set[str] = set()
		for item in response.get('items', []):
				status = item.get('status', {})
				content_details = item.get('contentDetails', {})
				if status.get('privacyStatus') != 'public':
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


def repair_tsv_file(split: str, category: str, videos_filenames: set[str]) -> None:
		pending: list[str] = []
		blacklist: set[str] = set()
		video_ids_updated = videos_filenames
		quota: int = DESIRED_VIDEO_PER_CATEGORY[split]

		def flush_batch() -> None:
				nonlocal pending
				sanitized_pending = are_videos_available(pending)
				for video in pending:
						if video in sanitized_pending and len(video_ids_updated) < quota:
								video_ids_updated.add(video)
				pending = []

		with open(f'{DATA_PATH}/{split}/urls/{category}.tsv', 'r', encoding='utf-8') as file:
			reader = csv.reader(file, delimiter='\t')
			blacklist = {row[1] for row in reader if row and len(row) > 1 and row[1].strip()}

		for shard in iterate_shards(f'{DATA_PATH}/{split}', split):
				labels = set(shard['labels'].values.numpy())
				cat = find_macro_category(labels)
				if cat is not None and cat == category and len(video_ids_updated) < quota:
						video_info = yt8m_key_to_video(shard['id'].numpy().decode('utf-8'))
						if video_info is not None and video_info[1] not in blacklist:
							pending.append(video_info[1])
						if len(pending) >= YT_API_BATCH:
								flush_batch()
						if len(video_ids_updated) >= quota:
								break

		if len(video_ids_updated) == quota:
				export_urls_to_tsv(split, category, video_ids_updated)
				print(f'[{time.strftime("%H:%M:%S")} | {split}] TSV file regenerated for category: {category}')

def repair_downloads(split: str, category: str) -> None:
		success:	bool = True
		tsv_path = pathlib.Path(DATA_PATH) / split / 'urls' / f'{category}_reloaded.tsv'
		output_path = pathlib.Path(VIDEO_PATH) / split / category
		with tsv_path.open(newline='', encoding='utf-8') as file:
				reader = csv.reader(file, delimiter='\t')
				urls = {row[1].strip() for row in reader if row and row[1].strip()}
		for url in urls:
				success = success and download_and_clip(url, output_path)
				if not success:
						print(f'[{time.strftime("%H:%M:%S")}] Error processing TSV file: {tsv_path}')

def repair_dataset() -> None:
		build_categories_from_csv_dict()
		for split in ['train', 'validation', 'test']:
				split_video_path = f'{VIDEO_PATH}/{split}'
				wanted_categories: set[str] = sel_m_categories.copy()
				if len(wanted_categories)	== 0:
						wanted_categories = set([folder.name for folder in pathlib.Path(split_video_path).iterdir() if folder.is_dir()])
				for category in wanted_categories:
						category_video_path = f'{split_video_path}/{category}'
						downloaded_videos =	list(pathlib.Path(category_video_path).glob('*'))
						if len(downloaded_videos) < DESIRED_VIDEO_PER_CATEGORY[split]:
								print(f'[{time.strftime("%H:%M:%S")} | {split}] Repairing category: {category} - '
								f'{len(downloaded_videos)}/{DESIRED_VIDEO_PER_CATEGORY[split]} videos found')
								repair_tsv_file(split, category, set([video.stem for video in downloaded_videos]))
								repair_downloads(split, category)
								print(f'[{time.strftime("%H:%M:%S")} | {split}] Repaired category: {category}')

if __name__ == "__main__":
	repair_dataset()
	exit(0)
