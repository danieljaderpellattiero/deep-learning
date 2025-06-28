from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score)

import umap
import h5py
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

EMBEDDING_PATH = '/home/hwanseok/data'

def load_class_embeddings() -> Dict[str, np.ndarray]:
    classes = {}
    with h5py.File(f'{EMBEDDING_PATH}/classes.hdf5', 'r') as file:
        for class_name in file['classes_embeddings']:
            classes[class_name] = file['classes_embeddings'][class_name][:]
    return classes

def load_video_text_embedding(file_path: pathlib.Path) -> np.ndarray or None:
    try:
        with h5py.File(file_path, 'r') as file:
            return file['embedded_features']['video_text_features'][:]
    except Exception as e:
        print(f"Failed to load {file_path.name}: {e}")
        return None

def classify_video(video_embedding: np.ndarray, class_embeddings: Dict[str, np.ndarray]) -> str:
    similarities = {
        class_name: cosine_similarity(video_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0]
        for class_name, emb in class_embeddings.items()
    }
    return max(similarities, key=similarities.get)

def plot_improved_confusion_matrix(y_true: List[str], y_pred: List[str], 
                                 split_name: str = "", save_path: str = None):
    """개선된 confusion matrix 시각화"""
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 더 큰 figure 크기와 개선된 레이아웃
    plt.figure(figsize=(14, 12))
    
    # seaborn을 사용한 더 나은 시각화
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    # 히트맵 생성
    ax = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', 
                     cbar_kws={'shrink': 0.8}, square=True,
                     linewidths=0.5, linecolor='white')
    
    # 라벨 회전 및 정렬 개선
    ax.set_xlabel('Predicted label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True label', fontsize=12, fontweight='bold')
    
    title = f'Confusion Matrix - {split_name}' if split_name else 'Confusion Matrix'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # x축 라벨 회전 및 정렬
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # 레이아웃 조정
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.15)
    
    if save_path is None:
        save_path = f"confusion_matrix_{split_name.lower()}.png" if split_name else "confusion_matrix.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def create_umap_visualization(embeddings: np.ndarray, labels: List[str], 
                            split_name: str = "", save_path: str = None):
    """UMAP을 사용한 임베딩 시각화"""
    print(f"Creating UMAP visualization for {split_name if split_name else 'all data'}...")
    
    # UMAP 차원 축소
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, 
                       metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # 고유한 라벨과 색상 매핑
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # 시각화
    plt.figure(figsize=(12, 10))
    
    # 각 클래스별로 점 그리기
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   c=[label_to_color[label]], label=label, alpha=0.6, s=30)
    
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    
    title = f'UMAP Visualization - {split_name}' if split_name else 'UMAP Visualization - All Data'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # 범례 설정
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
              fontsize=10, markerscale=1.5)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"umap_{split_name.lower()}.png" if split_name else "umap_all.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"UMAP visualization saved to: {save_path}")

def calculate_comprehensive_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """멀티클래스 분류를 위한 핵심 성능 지표 계산"""
    # 기본 정확도
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1-score (macro 평균)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

def create_performance_report(y_true: List[str], y_pred: List[str], 
                            split_name: str = "", save_path: str = None):
    """상세한 성능 리포트 생성"""
    labels = sorted(set(y_true + y_pred))
    
    # 핵심 지표 계산
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    
    # 클래스별 상세 리포트
    class_report = classification_report(y_true, y_pred, target_names=labels, 
                                       output_dict=True, zero_division=0)
    
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append(f"PERFORMANCE REPORT - {split_name if split_name else 'OVERALL'}")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # 전체 성능 지표
    report_lines.append("OVERALL METRICS:")
    report_lines.append("-" * 30)
    report_lines.append(f"Accuracy:       {metrics['accuracy']:.4f}")
    report_lines.append(f"Precision:      {metrics['precision_macro']:.4f}")
    report_lines.append(f"Recall:         {metrics['recall_macro']:.4f}")
    report_lines.append(f"F1-Score:       {metrics['f1_macro']:.4f}")
    report_lines.append("")
    
    # 클래스별 성능
    report_lines.append("CLASS-WISE PERFORMANCE:")
    report_lines.append("-" * 50)
    report_lines.append(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    report_lines.append("-" * 50)
    
    for class_name in labels:
        if class_name in class_report:
            precision = class_report[class_name]['precision']
            recall = class_report[class_name]['recall']
            f1 = class_report[class_name]['f1-score']
            report_lines.append(f"{class_name:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    # 파일로 저장
    if save_path is None:
        save_path = f"performance_report_{split_name.lower()}.txt" if split_name else "performance_report.txt"
    
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return metrics, report_lines

def collect_embeddings_and_labels(split: str, class_embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], List[str]]:
    """특정 split의 모든 임베딩과 라벨 수집"""
    embeddings_list = []
    true_labels = []
    pred_labels = []
    
    split_path = pathlib.Path(f'{EMBEDDING_PATH}/{split}')
    if not split_path.exists():
        return None, None, None
    
    for category_dir in split_path.iterdir():
        if not category_dir.is_dir():
            continue
        true_label = category_dir.name
        
        for file in category_dir.glob('*.hdf5'):
            video_embedding = load_video_text_embedding(file)
            if video_embedding is None:
                continue
            
            embeddings_list.append(video_embedding.flatten())
            true_labels.append(true_label)
            
            # 예측 라벨 계산
            pred = classify_video(video_embedding, class_embeddings)
            pred_labels.append(pred)
    
    if embeddings_list:
        embeddings_array = np.vstack(embeddings_list)
        return embeddings_array, true_labels, pred_labels
    else:
        return None, None, None

def evaluate_all():
    class_embeddings = load_class_embeddings()
    
    # 전체 데이터를 위한 리스트
    all_embeddings = []
    all_true_labels = []
    all_pred_labels = []
    
    # split별 결과 저장
    split_results = {}
    
    # 각 split 처리
    for split in ['train', 'validation', 'test']:
        print(f"\nProcessing {split} split...")
        
        embeddings, true_labels, pred_labels = collect_embeddings_and_labels(split, class_embeddings)
        
        if embeddings is None:
            print(f"No data found for {split} split")
            continue
        
        # 전체 데이터에 추가
        all_embeddings.append(embeddings)
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
        
        # split별 결과 저장
        split_results[split] = {
            'embeddings': embeddings,
            'true_labels': true_labels,
            'pred_labels': pred_labels
        }
        
        # split별 정확도 계산
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        total = len(true_labels)
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{split} accuracy: {correct}/{total} = {accuracy:.2f}%")
        
        # split별 confusion matrix 생성
        plot_improved_confusion_matrix(true_labels, pred_labels, split_name=split.capitalize())
        
        # split별 성능 리포트 생성
        metrics, _ = create_performance_report(true_labels, pred_labels, split_name=split.capitalize())
        
        # split별 UMAP 시각화
        create_umap_visualization(embeddings, true_labels, split_name=split.capitalize())
    
    # 전체 데이터에 대한 분석
    if all_embeddings:
        print("\nProcessing all data combined...")
        all_embeddings_array = np.vstack(all_embeddings)
        
        # 전체 confusion matrix
        plot_improved_confusion_matrix(all_true_labels, all_pred_labels, split_name="All")
        
        # 전체 성능 리포트
        metrics, _ = create_performance_report(all_true_labels, all_pred_labels, split_name="All")
        
        # 전체 UMAP 시각화
        create_umap_visualization(all_embeddings_array, all_true_labels, split_name="All")
        
        # 전체 정확도
        total_correct = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == p)
        total_count = len(all_true_labels)
        overall_acc = (total_correct / total_count * 100) if total_count > 0 else 0
        
        print("\n" + "="*40)
        print("OVERALL PERFORMANCE SUMMARY")
        print("="*40)
        print(f"Total Accuracy:  {overall_acc:.2f}%")
        print(f"Precision:       {metrics['precision_macro']:.4f}")
        print(f"Recall:          {metrics['recall_macro']:.4f}")
        print(f"F1-Score:        {metrics['f1_macro']:.4f}")
        print("="*40)
    
    print("\nAll visualizations and reports have been generated!")

if __name__ == "__main__":
    evaluate_all()
