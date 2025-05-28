import time
import h5py
import ffmpeg
import pathlib
import threading
import numpy as np
from typing import Any
from twelvelabs import TwelveLabs
from openai import OpenAI, OpenAIError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from twelvelabs.models.embed import SegmentEmbedding
from concurrent.futures import ProcessPoolExecutor, as_completed
# --------------------------------------------------------------------------- #
#  API KEYS AND PATHS                                                         #
# --------------------------------------------------------------------------- #
TWELVE_LABS_API_KEY = ''
YOUTUBE_API_KEY = ''
OPEN_AI_API_KEY = ''
# --------------------------------------------------------------------------- #
#  CONSTANTS                                                                  #
# --------------------------------------------------------------------------- #
YT_API_BATCH = 50
VIDEO_PATH = './videos'
EMBEDDING_PATH = './embeddings'
# --------------------------------------------------------------------------- #
#  CONFIGURATION                                                              #
# --------------------------------------------------------------------------- #
hdf5_lock = threading.Lock()
openai_client = OpenAI(api_key=OPEN_AI_API_KEY)
twelvelabs_client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)
youtube_client = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
# --------------------------------------------------------------------------- #
#  VIDEO MACRO CATEGORIES                                                     #
# --------------------------------------------------------------------------- #
categories: list[str] = []

def init_categories() -> bool:
	"""
	Initializes the categories from the video dataset directory.
	"""
	global categories
	categories = [folder.name for folder in pathlib.Path(f'{VIDEO_PATH}/train').iterdir() if folder.is_dir()]
	if not categories:
		print(f'[{time.strftime("%H:%M:%S")}] No categories found in {VIDEO_PATH}/train')
		return False
	return True

def init_folders() -> None:
	"""
	Initializes the necessary folders for storing embeddings and video splits.
	"""
	if not pathlib.Path(EMBEDDING_PATH).exists():
		pathlib.Path(EMBEDDING_PATH).mkdir(parents=True, exist_ok=True)
	for dset_split in ['train', 'validation', 'test']:
		dset_path = pathlib.Path(f'{VIDEO_PATH}/{dset_split}')
		if not dset_path.exists():
			pathlib.Path(dset_path).mkdir(parents=True, exist_ok=True)
		for category in categories:
			category_path = pathlib.Path(f'{EMBEDDING_PATH}/{dset_split}/{category}')
			if not category_path.exists():
				pathlib.Path(category_path).mkdir(parents=True, exist_ok=True)

def iso8601_to_seconds(iso_duration: str) -> int:
	"""
	Converts an ISO 8601 duration string to seconds.

	:param iso_duration: ISO 8601 duration string (e.g., "PT1H30M45S")
	:return: duration in seconds
	"""
	hours: int = 0
	minutes: int = 0
	seconds: int = 0
	number: str = ''
	for char in iso_duration.lstrip('PT'):
		if char.isdigit():
			number += char
		elif char == 'H':
			hours = int(number) if number else 0
			number = ''
		elif char == 'M':
			minutes = int(number) if number else 0
			number = ''
		elif char == 'S':
			seconds = int(number) if number else 0
			number = ''
	return hours * 3600 + minutes * 60 + seconds

def normalize_metadata(metadata: dict[str, Any]) -> np.ndarray:
	"""
	Normalizes numerical metadata by converting it to a numpy array, normalizing it.

	:param metadata: dictionary containing metadata with numerical values
	:return: normalized numpy array of numerical metadata
	"""
	numerical_metadata: list[float] = []
	for key, value in metadata.items():
		if isinstance(value, int | float):
			numerical_metadata.append(float(value))
	return np.array(numerical_metadata, dtype=np.float32) / np.linalg.norm(numerical_metadata) + np.finfo(np.float32).eps

def stringify_metadata(metadata: dict[str, Any]) -> str:
	"""
	Converts metadata dictionary to a single string representation. (for OpenAI embedding)

	:param metadata: dictionary containing metadata
	:return: string representation of the metadata
	"""
	string_metadata: str = ''
	for key, value in metadata.items():
		if isinstance(value, list):
			string_metadata += f'{key}:{", ".join(value)};'
		if isinstance(value, str):
			string_metadata += f'{key}:{value};'
	return string_metadata.strip()[:-1]

def embed_metadata(metadata: dict[str, Any]) -> np.ndarray or None:
	"""
	Embeds metadata using OpenAI's text-embedding-3-large model.

	:param metadata: dictionary containing metadata to be embedded
	:return: numpy array of embedded metadata or None if an error occurs
	"""
	str_metadata: str = stringify_metadata(metadata)
	try:
		response = openai_client.embeddings.create(
			input=str_metadata,
			model='text-embedding-3-large'
		)
		embedded_metadata = np.array(response.data[0].embedding, dtype=np.float32)
		return embedded_metadata
	except OpenAIError as e:
		print(f'[{time.strftime("%H:%M:%S")}] OpenAI API error: {e}')
		return None

def fetch_ffmpeg_metadata(video_path: pathlib.Path) -> dict[str, Any]:
	"""
	Fetches technical metadata from a video file using ffmpeg.

	:param video_path: path to the video file
	:return: dictionary containing video and audio codec, resolution, fps, sample rate, and channels
	"""
	tech_specs = ffmpeg.probe(video_path)
	video_stream = [stream for stream in tech_specs['streams'] if stream['codec_type'] == 'video']
	audio_stream = [stream for stream in tech_specs['streams'] if stream['codec_type'] == 'audio']
	metadata: dict[str, Any] = {
		'video_codec': video_stream[0]['codec_name'] if video_stream else 'unknown',
		'width': video_stream[0]['width'] if video_stream else 0,
		'height': video_stream[0]['height'] if video_stream else 0,
		'fps': eval(video_stream[0]['avg_frame_rate']) if video_stream else 0,
		'audio_codec': audio_stream[0]['codec_name'] if audio_stream else 'unknown',
		'sample_rate': int(audio_stream[0]['sample_rate']) if audio_stream else 0,
		'channels': audio_stream[0].get('channels', 0) if audio_stream else 0,
	}
	return metadata

def fetch_youtube_metadata(video_id: str) -> dict[str, Any] or None:
	"""
	Fetches metadata for a YouTube video using the YouTube Data API v3.

	:param video_id: YouTube video ID (11-character string)
	:return: dictionary containing video metadata or None if an error occurs
	"""
	try:
		request = youtube_client.videos().list(part='snippet,contentDetails,statistics,status', id=video_id)
		response = request.execute()
		item = response.get('items', [None])[0]
		if item is not None:
			snippet = item.get('snippet', {})
			statistics = item.get('statistics', {})
			content_details = item.get('contentDetails', {})
			metadata: dict[str, Any] = {
				'tags': snippet.get('tags', []),
				'published_at': snippet.get('publishedAt', ''),
				'channel_title': snippet.get('channelTitle', ''),
				'default_language': snippet.get('defaultLanguage', 'unknown') or snippet.get('defaultAudioLanguage', 'unknown'),
				'view_count': int(statistics.get('viewCount', 0)),
				'like_count': int(statistics.get('likeCount', 0)),
				'comment_count': int(statistics.get('commentCount', 0)),
				'favorite_count': int(statistics.get('favoriteCount', 0)),
				'dimension': content_details.get('dimension', 'unknown'),
				'definition': content_details.get('definition', 'unknown'),
				'projection': content_details.get('projection', 'unknown'),
				'duration_sec': iso8601_to_seconds(content_details.get('duration', 'PT0S')),
				'rating': content_details.get('contentRating', {}).get('kmrbRating', 'unknown')
			}
			return metadata
		return None
	except HttpError as e:
		print(f'[{time.strftime("%H:%M:%S")}] YouTube API error: {e}')
		return None

def fetch_twelvelabs_embeddings(video_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray] or None:
	"""
	Fetches video and audio embeddings from TwelveLabs API for a given video file.

	:param video_path: path to the video file
	:return: tuple of numpy arrays containing video text features and audio features, or None if an error occurs
	"""
	try:
		task = twelvelabs_client.embed.task.create(model_name='Marengo-retrieval-2.7', video_file=str(video_path))
		task.wait_for_done(sleep_interval=1)
		if task.status != 'ready':
			print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: Task status is {task.status}, expected "ready"')
			return None
		task = task.retrieve(embedding_option=['visual-text', 'audio'])
		if not task.video_embedding:
			print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: No video embedding found in task')
			return None
		if not task.video_embedding.segments:
			print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: No segments found in video embedding')
			return None
		periods: set[float] = {0.0}
		multimodal_embeddings: dict[str, list[np.ndarray]] = {
			'visual-text': [],
			'audio': []
		}
		for segment in task.video_embedding.segments:
			periods.add(segment.end_offset_sec)
			multimodal_embeddings[segment.embedding_option].append(np.array(segment.embeddings_float))
		video_features: np.ndarray = np.mean(np.array(multimodal_embeddings['visual-text']), axis=0)
		audio_features: np.ndarray = np.mean(np.array(multimodal_embeddings['audio']), axis=0)
		return video_features, audio_features
	except Exception as e:
		print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: {e}')
		return None

def export_video_features(dset_split: str,	video_category: str, video_id: str,
                          ffmpeg_mdata: dict[str, Any], youtube_mdata: dict[str, Any],
                          normalized_num_mdata: np.ndarray, embedded_str_mdata: np.ndarray,
                          video_text_features: np.ndarray, audio_features: np.ndarray) -> None:
	"""
	Exports video features to an HDF5 file.

	:param dset_split: dataset split (train, validation, test)
	:param video_category: video category (e.g., 'sports', 'music')
	:param video_id: unique identifier for the video
	:param ffmpeg_mdata: dictionary containing ffmpeg metadata
	:param youtube_mdata: dictionary containing YouTube metadata
	:param normalized_num_mdata: numpy array of normalized numerical metadata
	:param embedded_str_mdata: numpy array of embedded string metadata
	:param video_text_features: numpy array of video text features from TwelveLabs
	:param audio_features: numpy array of audio features from TwelveLabs
	"""
	with hdf5_lock:
		with h5py.File(f'{EMBEDDING_PATH}/{dset_split}/{video_category}/{video_id}.hdf5', 'w') as file:
			raw_features = file.create_group('raw_features')
			ffmpeg_numerical = np.array([
				ffmpeg_mdata['width'],
				ffmpeg_mdata['height'],
				ffmpeg_mdata['fps'],
				ffmpeg_mdata['sample_rate'],
				ffmpeg_mdata['channels']
			])
			raw_features.create_dataset('ffmpeg_numeric', data=ffmpeg_numerical, dtype=np.float32)
			raw_features.create_dataset('video_codec', data=ffmpeg_mdata['video_codec'].encode('utf-8'),
			                            dtype=h5py.string_dtype())
			raw_features.create_dataset('audio_codec', data=ffmpeg_mdata['audio_codec'].encode('utf-8'),
			                            dtype=h5py.string_dtype())
			youtube_numerical = np.array([
				youtube_mdata['view_count'],
				youtube_mdata['like_count'],
				youtube_mdata['comment_count'],
				youtube_mdata['favorite_count'],
				youtube_mdata['duration_sec']
			])
			raw_features.create_dataset('youtube_numerical', data=youtube_numerical, dtype=np.float32)
			raw_features.create_dataset('youtube_tags', data=youtube_mdata['tags'], dtype=h5py.string_dtype(encoding='utf-8'))
			raw_features.create_dataset('youtube_rating', data=youtube_mdata['rating'].encode('utf-8'),
			                            dtype=h5py.string_dtype())
			raw_features.create_dataset('youtube_dimension', data=youtube_mdata['dimension'].encode('utf-8'),
			                            dtype=h5py.string_dtype())
			raw_features.create_dataset('youtube_definition', data=youtube_mdata['definition'].encode('utf-8'),
			                            dtype=h5py.string_dtype())
			raw_features.create_dataset('youtube_projection', data=youtube_mdata['projection'].encode('utf-8'),
			                            dtype=h5py.string_dtype())
			raw_features.create_dataset('youtube_published_at', data=youtube_mdata['published_at'].encode('utf-8'),
			                            dtype=h5py.string_dtype())
			raw_features.create_dataset('youtube_channel_title', data=youtube_mdata['channel_title'].encode('utf-8'),
			                            dtype=h5py.string_dtype())
			raw_features.create_dataset('youtube_default_language', data=youtube_mdata['default_language'].encode('utf-8'),
			                            dtype=h5py.string_dtype())

			embedded_features = file.create_group('embedded_features')
			embedded_features.create_dataset('audio_features', data=audio_features, dtype=np.float32)
			embedded_features.create_dataset('video_text_features', data=video_text_features, dtype=np.float32)
			embedded_features.create_dataset('embedded_string_metadata', data=embedded_str_mdata, dtype=np.float32)
			embedded_features.create_dataset('normalized_numerical_metadata', data=normalized_num_mdata, dtype=np.float32)

def load_video_features(video_split: str, video_category: str, video_id: str) -> dict[str, Any] or None:
	"""
	Loads video features from an HDF5 file. (only for debugging purposes)
	"""
	video_features: dict[str, Any] = {}
	video_features_path = pathlib.Path(f'{EMBEDDING_PATH}/{video_split}/{video_category}/{video_id}.hdf5')
	if video_features_path.exists():
		with h5py.File(video_features_path, 'r') as file:
			raw_features = file['raw_features']
			video_features['ffmpeg_numerical'] = raw_features['ffmpeg_numerical'][:]
			video_features['youtube_tags'] = raw_features['youtube_tags'][:].tolist()
			video_features['youtube_numerical'] = raw_features['youtube_numerical'][:]
			video_features['video_codec'] = raw_features['video_codec'][()].decode('utf-8')
			video_features['audio_codec'] = raw_features['audio_codec'][()].decode('utf-8')
			video_features['youtube_rating'] = raw_features['youtube_rating'][()].decode('utf-8')
			video_features['youtube_dimension'] = raw_features['youtube_dimension'][()].decode('utf-8')
			video_features['youtube_definition'] = raw_features['youtube_definition'][()].decode('utf-8')
			video_features['youtube_projection'] = raw_features['youtube_projection'][()].decode('utf-8')
			video_features['youtube_published_at'] = raw_features['youtube_published_at'][()].decode('utf-8')
			video_features['youtube_channel_title'] = raw_features['youtube_channel_title'][()].decode('utf-8')
			video_features['youtube_default_language'] = raw_features['youtube_default_language'][()].decode('utf-8')
			embedded_features = file['embedded_features']
			video_features['audio_features'] = embedded_features['audio_features'][:]
			video_features['video_text_features'] = embedded_features['video_text_features'][:]
			video_features['embedded_string_metadata'] = embedded_features['embedded_string_metadata'][:]
			video_features['normalized_numerical_metadata'] = embedded_features['normalized_numerical_metadata'][:]
		return video_features
	return None

def fetch_class_embeddings() -> bool:
	"""
	Fetches class embeddings for each category using TwelveLabs API and saves them to an HDF5 file.

	:return: True if embeddings are successfully fetched and saved, False otherwise
	"""
	classes: dict[str, np.ndarray] = {}
	if not pathlib.Path(f'{EMBEDDING_PATH}/classes.hdf5').exists():
		for category in categories:
			try:
				embedding = twelvelabs_client.embed.create(model_name='Marengo-retrieval-2.7', text=category)
				text_embedding = getattr(embedding, 'text_embedding', None)
				if text_embedding is None or text_embedding.segments is None or not text_embedding.segments:
					print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: No text embedding found for class: {category}')
					return False
				if not isinstance(embedding.text_embedding.segments[0], SegmentEmbedding):
					print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: Segment is not a SegmentEmbedding for class: '
					      f'{category}')
					return False
				classes.setdefault(category, np.array(embedding.text_embedding.segments[0].embeddings_float))
			except Exception as e:
				print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: {e} for class: {category}')
				return False
		with h5py.File(f'{EMBEDDING_PATH}/classes.hdf5', 'w') as file:
			classes_embedding_group = file.create_group('classes_embeddings')
			for class_name, embedding in classes.items():
				classes_embedding_group.create_dataset(class_name, data=embedding)
		print(f'[{time.strftime("%H:%M:%S")}] Classes embeddings generated and saved to {EMBEDDING_PATH}/classes.hdf5')
	else:
		print(f'[{time.strftime("%H:%M:%S")}] Classes embeddings already exist, skipping generation')
	return True

def load_class_embeddings() -> dict[str, np.ndarray] or None:
	"""
	Loads class embeddings from an HDF5 file. (only for debugging purposes)

	:return: dictionary of class names and their embeddings, or None if the file does not exist
	"""
	classes: dict[str, np.ndarray] = {}
	if pathlib.Path(f'{EMBEDDING_PATH}/classes.hdf5').exists():
		with h5py.File(f'{EMBEDDING_PATH}/classes.hdf5', 'r') as file:
			for class_name in file['classes_embeddings']:
				classes[class_name] = file['classes_embeddings'][class_name][:]
				print(f'[{time.strftime("%H:%M:%S")}] Class: {class_name}, Embedding: {classes[class_name]}')
		return classes
	return None

def generate_features(dset_split: str, category: str) -> None:
	"""
	Generates video features for a given dataset split and category.

	:param dset_split: dataset split (train, validation, test)
	:param category: video category (e.g., 'sports', 'music')
	"""
	processed_videos: int = 0
	processed_videos_ids: list[str] = [file.stem for file in
	                                   pathlib.Path(f'{EMBEDDING_PATH}/{dset_split}/{category}').iterdir()
	                                   if file.is_file()]
	video_ids: list[str] = [file.stem for file in
	                       pathlib.Path(f'{VIDEO_PATH}/{dset_split}/{category}').iterdir() if file.is_file()]
	for video_id in video_ids:
		video_url = pathlib.Path(f'{VIDEO_PATH}/{dset_split}/{category}/{video_id}.mp4')
		if video_id in processed_videos_ids:
			continue
		tl_features = fetch_twelvelabs_embeddings(video_url)
		if tl_features is None or tl_features[0] is None or tl_features[1] is None:
			continue
		yt_metadata = fetch_youtube_metadata(video_id)
		if yt_metadata is None:
			continue
		ffmpeg_metadata = fetch_ffmpeg_metadata(video_url)
		embedded_text_metadata = embed_metadata(yt_metadata | ffmpeg_metadata)
		if embedded_text_metadata is None:
			continue
		normalized_numerical_metadata = normalize_metadata(yt_metadata | ffmpeg_metadata)
		export_video_features(
			dset_split, category, video_id,
			ffmpeg_metadata, yt_metadata,
			normalized_numerical_metadata, embedded_text_metadata,
			tl_features[0], tl_features[1]
		)
		processed_videos += 1
	print(f'[{time.strftime("%H:%M:%S")} | {dset_split}, {category}] Processed {processed_videos}/'
	      f'{len(video_ids) - len(processed_videos_ids)} videos')
	return None

if __name__ == "__main__":
	if not init_categories():
		exit(1)
	init_folders()
	if not fetch_class_embeddings():
		exit(1)
	for split in ['train', 'validation', 'test']:
		# Process pool has been chosen over thread pool to avoid segmentation faults
		with ProcessPoolExecutor(max_workers=len(categories)) as executor:
			promises = [executor.submit(generate_features, split, category) for category in categories]
			for future in as_completed(promises):
				future.result()
	exit(0)
