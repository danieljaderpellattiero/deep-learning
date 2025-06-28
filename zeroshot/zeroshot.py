from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             precision_recall_fscore_support, accuracy_score)

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

def plot_improved_confusion_matrix(y_true: List[str], y_pred: List[str], save_path: str = "confusion_matrix.png"):
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
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # x축 라벨 회전 및 정렬
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # 레이아웃 조정
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.15)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Improved confusion matrix saved to: {save_path}")

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

def create_performance_report(y_true: List[str], y_pred: List[str], save_path: str = "performance_report.txt"):
    """상세한 성능 리포트 생성"""
    labels = sorted(set(y_true + y_pred))
    
    # 핵심 지표 계산
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    
    # 클래스별 상세 리포트
    class_report = classification_report(y_true, y_pred, target_names=labels, 
                                       output_dict=True, zero_division=0)
    
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append("PERFORMANCE REPORT")
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
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return metrics, report_lines

def evaluate_all():
    class_embeddings = load_class_embeddings()
    total_correct, total_count = 0, 0
    per_split_results = {}

    all_true_labels: List[str] = []
    all_pred_labels: List[str] = []

    # 결과 저장용 리스트
    summary_lines = []

    for split in ['train', 'validation', 'test']:
        split_path = pathlib.Path(f'{EMBEDDING_PATH}/{split}')
        if not split_path.exists():
            continue

        split_total, split_correct = 0, 0

        for category_dir in split_path.iterdir():
            if not category_dir.is_dir():
                continue
            true_label = category_dir.name

            for file in category_dir.glob('*.hdf5'):
                video_embedding = load_video_text_embedding(file)
                if video_embedding is None:
                    continue

                pred = classify_video(video_embedding, class_embeddings)

                all_true_labels.append(true_label)
                all_pred_labels.append(pred)

                if pred == true_label:
                    split_correct += 1
                split_total += 1

        acc = (split_correct / split_total * 100) if split_total > 0 else 0
        per_split_results[split] = (split_correct, split_total, acc)
        total_correct += split_correct
        total_count += split_total
        summary_lines.append(f"{split.capitalize():<10} | Accuracy: {split_correct}/{split_total} = {acc:.2f}%")

    # 전체 정확도 추가
    overall_acc = (total_correct / total_count * 100) if total_count > 0 else 0
    summary_lines.append(f"\nOverall Accuracy: {total_correct}/{total_count} = {overall_acc:.2f}%")

    # 기본 요약 텍스트 파일로 저장
    summary_path = pathlib.Path("evaluation_summary.txt")
    with summary_path.open("w") as f:
        f.write("=== Evaluation Summary ===\n")
        f.write("\n".join(summary_lines))
    print(f"Evaluation summary saved to: {summary_path.resolve()}")

    # 개선된 Confusion Matrix 생성
    plot_improved_confusion_matrix(all_true_labels, all_pred_labels)
    
    # 포괄적인 성능 리포트 생성
    metrics, report_lines = create_performance_report(all_true_labels, all_pred_labels)
    
    # 콘솔에 핵심 지표 출력
    print("\n" + "="*40)
    print("PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision:       {metrics['precision_macro']:.4f}")
    print(f"Recall:          {metrics['recall_macro']:.4f}")
    print(f"F1-Score:        {metrics['f1_macro']:.4f}")
    print("="*40)
    
    print("\nDetailed performance report saved to: performance_report.txt")

if __name__ == "__main__":
    evaluate_all()
