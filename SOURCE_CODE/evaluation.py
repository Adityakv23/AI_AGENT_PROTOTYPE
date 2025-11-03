"""
Evaluation metrics for assessing AI agent performance
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_compute
import torch

from config import TEST_DATA_PATH, EVALUATION_CONFIG


class Evaluator:
    """
    Comprehensive evaluation system for measuring agent quality
    """
    
    def __init__(self):
        self.evaluation_config = EVALUATION_CONFIG
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test dataset"""
        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compute_rouge_scores(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores (precision, recall, F1)"""
        rouge_scores = {
            'rouge1': {'precision': [], 'recall': [], 'f1': []},
            'rouge2': {'precision': [], 'recall': [], 'f1': []},
            'rougeL': {'precision': [], 'recall': [], 'f1': []}
        }
        
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                rouge_scores[metric]['precision'].append(scores[metric].precision)
                rouge_scores[metric]['recall'].append(scores[metric].recall)
                rouge_scores[metric]['f1'].append(scores[metric].fmeasure)
        
        # Average scores
        avg_scores = {}
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            avg_scores[f'{metric}_precision'] = np.mean(rouge_scores[metric]['precision'])
            avg_scores[f'{metric}_recall'] = np.mean(rouge_scores[metric]['recall'])
            avg_scores[f'{metric}_f1'] = np.mean(rouge_scores[metric]['f1'])
        
        return avg_scores
    
    def compute_bert_score(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Compute BERTScore for semantic similarity"""
        P, R, F1 = bert_score_compute(
            predictions, 
            references, 
            lang='en',
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        return {
            'bert_score_precision': P.mean().item(),
            'bert_score_recall': R.mean().item(),
            'bert_score_f1': F1.mean().item()
        }
    
    def compute_length_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Compute compression ratio and length statistics"""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        compression_ratios = [
            p_len / ref_len if ref_len > 0 else 0
            for p_len, ref_len in zip(pred_lengths, ref_lengths)
        ]
        
        return {
            'avg_compression_ratio': np.mean(compression_ratios),
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'std_pred_length': np.std(pred_lengths)
        }
    
    def evaluate_generation(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation combining multiple metrics
        """
        print("\n=== Evaluation Metrics ===\n")
        
        # ROUGE scores
        print("Computing ROUGE scores...")
        rouge_scores = self.compute_rouge_scores(predictions, references)
        
        # BERTScore
        print("Computing BERTScore...")
        bert_scores = self.compute_bert_score(predictions, references)
        
        # Length metrics
        print("Computing length metrics...")
        length_metrics = self.compute_length_metrics(predictions, references)
        
        # Combine all metrics
        all_metrics = {
            **rouge_scores,
            **bert_scores,
            **length_metrics
        }
        
        # Print results
        print("\n--- Results ---")
        print(f"\nROUGE-1 F1: {all_metrics['rouge1_f1']:.4f}")
        print(f"ROUGE-2 F1: {all_metrics['rouge2_f1']:.4f}")
        print(f"ROUGE-L F1: {all_metrics['rougeL_f1']:.4f}")
        print(f"\nBERTScore F1: {all_metrics['bert_score_f1']:.4f}")
        print(f"BERTScore Precision: {all_metrics['bert_score_precision']:.4f}")
        print(f"BERTScore Recall: {all_metrics['bert_score_recall']:.4f}")
        print(f"\nAvg Compression Ratio: {all_metrics['avg_compression_ratio']:.2f}x")
        print(f"Avg Summary Length: {all_metrics['avg_pred_length']:.1f} words")
        
        return all_metrics
    
    def compare_models(
        self, 
        predictions_baseline: List[str],
        predictions_finetuned: List[str],
        references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compare baseline vs fine-tuned model"""
        print("\n=== Model Comparison ===\n")
        
        baseline_metrics = self.evaluate_generation(predictions_baseline, references)
        finetuned_metrics = self.evaluate_generation(predictions_finetuned, references)
        
        # Calculate improvements
        print("\n--- Improvements ---")
        improvements = {}
        for key in baseline_metrics:
            if isinstance(baseline_metrics[key], float):
                improvement = finetuned_metrics[key] - baseline_metrics[key]
                improvements[key] = improvement
                print(f"{key}: {improvement:+.4f} ({improvement/baseline_metrics[key]*100:+.1f}%)")
        
        return {
            'baseline': baseline_metrics,
            'finetuned': finetuned_metrics,
            'improvements': improvements
        }
    
    def generate_evaluation_report(
        self, 
        metrics: Dict[str, float],
        output_path: Path
    ) -> None:
        """Generate a detailed evaluation report"""
        report = {
            'evaluation_metrics': metrics,
            'methodology': {
                'rouge': 'ROUGE-n measures n-gram overlap between predicted and reference summaries',
                'bert_score': 'BERTScore measures semantic similarity using contextual embeddings',
                'compression_ratio': 'Ratio of summary length to original text length'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Evaluation report saved to: {output_path}")


# Example usage for testing without actual model
def generate_baseline_predictions(texts: List[str]) -> List[str]:
    """Generate baseline predictions (simple word-based extraction)"""
    predictions = []
    for text in texts:
        # Simple baseline: take first and last sentences
        sentences = text.split('. ')
        if len(sentences) >= 4:
            summary = '. '.join([sentences[0], sentences[-1]]) + '.'
        else:
            summary = text[:200] + '...'
        predictions.append(summary)
    return predictions


if __name__ == "__main__":
    # Test evaluation system
    evaluator = Evaluator()
    test_data = evaluator.load_test_data()
    
    # Generate baseline predictions
    texts = [item['input'] for item in test_data]
    references = [item['output'] for item in test_data]
    baseline_preds = generate_baseline_predictions(texts)
    
    # Evaluate
    metrics = evaluator.evaluate_generation(baseline_preds, references)
    
    # Save report
    from config import OUTPUT_DIR
    evaluator.generate_evaluation_report(metrics, OUTPUT_DIR / "baseline_evaluation.json")

