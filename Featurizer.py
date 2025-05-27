import time
import h5py
import ffmpeg
import pathlib
import numpy as np
from typing import Any
from twelvelabs import TwelveLabs
from openai import OpenAI, OpenAIError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from twelvelabs.models.embed import SegmentEmbedding

TWELVE_LABS_API_KEY = ''
YOUTUBE_API_KEY = ''
OPEN_AI_API_KEY = ''

YT_API_BATCH = 50
VIDEO_PATH = './videos'
EMBEDDING_PATH = './embeddings'
HARDCODED_VIDEO_PATH = './videos/train/Games/2TL2Dob88y8.mp4' # !

openai_client = OpenAI(api_key=OPEN_AI_API_KEY)
twelvelabs_client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)
youtube_client = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def iso8601_to_seconds(iso_duration: str) -> int:
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
	numerical_metadata: list[float] = []
	for key, value in metadata.items():
		if isinstance(value, int | float):
			numerical_metadata.append(float(value))
	return np.array(numerical_metadata, dtype=np.float32) / np.linalg.norm(numerical_metadata) + np.finfo(np.float32).eps

def stringify_metadata(metadata: dict[str, Any]) -> str:
	string_metadata: str = ''
	for key, value in metadata.items():
		if isinstance(value, list):
			string_metadata += f'{key}:{", ".join(value)};'
		if isinstance(value, str):
			string_metadata += f'{key}:{value};'
	return string_metadata.strip()[:-1]

def fetch_ffmpeg_metadata(video_path: pathlib.Path) -> dict[str, Any]:
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

def embed_metadata(metadata: dict[str, Any]) -> np.ndarray or None:
	str_metadata: str = stringify_metadata(metadata)
	embedded_metadata: np.ndarray or None = None
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

def fetch_twelvelabs_embeddings(video_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray] or None:
	try:
		task = twelvelabs_client.embed.task.create(model_name='Marengo-retrieval-2.7', video_file=str(video_path))
		task.wait_for_done(sleep_interval=1)
		if task.status != 'ready':
			print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: Task status is {task.status}, expected "ready".')
			return None
		task = task.retrieve(embedding_option=['visual-text', 'audio'])
		if not task.video_embedding:
			print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: No video embedding found in task.')
			return None
		if not task.video_embedding.segments:
			print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: No segments found in video embedding.')
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

def export_video_features(
	split: str,
	video_id: str,
	video_category: str,
	audio_features: np.ndarray,
	ffmpeg_mdata: dict[str, Any],
	youtube_mdata: dict[str, Any],
	video_text_features: np.ndarray,
	embedded_str_mdata: np.ndarray,
	normalized_num_mdata: np.ndarray,
) -> None:
	with h5py.File(f'{EMBEDDING_PATH}/{video_id}.hdf5', 'w') as file: #! make path
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

def load_video_features(split: str, video_category: str, video_id: str) -> dict[str, Any] or None:
	video_features: dict[str, Any] = {}
	video_features_path = pathlib.Path(f'{EMBEDDING_PATH}/{video_id}.hdf5')
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

def generate_class_embeddings() -> bool:
	classes: dict[str, np.ndarray] = {}
	if not pathlib.Path(EMBEDDING_PATH).exists():
		pathlib.Path(EMBEDDING_PATH).mkdir(parents=True, exist_ok=True)
	classification_categories = set([
		folder.name.replace('_', ' ')
		for folder in pathlib.Path(f'{VIDEO_PATH}/train').iterdir()
		if folder.is_dir()])
	if not pathlib.Path(f'{EMBEDDING_PATH}/classes.hdf5').exists():
		for category in classification_categories:
			try:
				embedding = twelvelabs_client.embed.create(model_name='Marengo-retrieval-2.7', text=category)
				text_embedding = getattr(embedding, 'text_embedding', None)
				if text_embedding is None or text_embedding.segments is None or not text_embedding.segments:
					print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: No text embedding found for class: {category}')
					return False
				if not isinstance(embedding.text_embedding.segments[0], SegmentEmbedding):
					print(f'[{time.strftime("%H:%M:%S")}] TwelveLabs API error: Segment is not a SegmentEmbedding for class: {category}')
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
	classes: dict[str, np.ndarray] = {}
	if pathlib.Path(f'{EMBEDDING_PATH}/classes.hdf5').exists():
		with h5py.File(f'{EMBEDDING_PATH}/classes.hdf5', 'r') as file:
			for class_name in file['classes_embeddings']:
				classes[class_name] = file['classes_embeddings'][class_name][:]
				print(f'[{time.strftime("%H:%M:%S")}] Class: {class_name}, Embedding: {classes[class_name]}')
		return classes
	return None

if __name__ == "__main__":
	if not generate_class_embeddings():
		exit(1)
	# Pipeline for a single video
	ffmpeg_metadata = fetch_ffmpeg_metadata(pathlib.Path(HARDCODED_VIDEO_PATH))
	youtube_metadata = fetch_youtube_metadata(pathlib.Path(HARDCODED_VIDEO_PATH).stem)
	if youtube_metadata is None:
		exit(1) # skipping
	normalized_numerical_metadata = normalize_metadata(youtube_metadata | ffmpeg_metadata)
	embedded_string_metadata = embed_metadata(youtube_metadata | ffmpeg_metadata)
	if embedded_string_metadata is None:
		exit(1) # skipping
	tl_video_features, tl_audio_features = fetch_twelvelabs_embeddings(pathlib.Path(HARDCODED_VIDEO_PATH))
	if tl_video_features is None or tl_audio_features is None:
		exit(1) # skipping
	export_video_features(
		split='train',
		video_id=pathlib.Path(HARDCODED_VIDEO_PATH).stem,
		video_category=pathlib.Path(HARDCODED_VIDEO_PATH).parent.name,
		audio_features=tl_audio_features,
		ffmpeg_mdata=ffmpeg_metadata,
		youtube_mdata=youtube_metadata,
		video_text_features=tl_video_features,
		embedded_str_mdata=embedded_string_metadata,
		normalized_num_mdata=normalized_numerical_metadata
	)
	# Pipeline for a single video
	features = load_video_features('train', pathlib.Path(HARDCODED_VIDEO_PATH).parent.name, pathlib.Path(HARDCODED_VIDEO_PATH).stem)
	for key, value in features.items():
		print(f'{key}: {value}')
	exit(0)
