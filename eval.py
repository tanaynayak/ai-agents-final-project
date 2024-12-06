import nltk
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score
from sentence_transformers import SentenceTransformer

# Download required NLTK data
nltk.download('wordnet', quiet=True)

# Load sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(file_path):
    """Load CSV file with error handling."""
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ["Q", "A"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

def calculate_semantic_similarity(questions, references, candidates):
    """Calculate semantic similarity using sentence embeddings."""
    # Combine question and answers for context
    reference_pairs = [f"{q} [SEP] {a}" for q, a in zip(questions, references)]
    candidate_pairs = [f"{q} [SEP] {c}" for q, c in zip(questions, candidates)]

    # Compute embeddings
    reference_embeddings = embedding_model.encode(reference_pairs)
    candidate_embeddings = embedding_model.encode(candidate_pairs)

    # Convert numpy arrays to torch tensors
    reference_embeddings = torch.tensor(reference_embeddings, dtype=torch.float32)
    candidate_embeddings = torch.tensor(candidate_embeddings, dtype=torch.float32)

    # Compute cosine similarity
    similarities = torch.nn.functional.cosine_similarity(candidate_embeddings, reference_embeddings)
    return similarities.mean().item()  # Convert tensor to Python float

def calculate_bleu(reference, candidate):
    """Calculate BLEU score with smoothing."""
    # Ensure inputs are strings and not NaN
    reference = str(reference) if pd.notna(reference) else ""
    candidate = str(candidate) if pd.notna(candidate) else ""

    smoother = SmoothingFunction().method1
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoother)

def calculate_rouge(reference, candidate):
    """Calculate ROUGE scores."""
    reference = str(reference) if pd.notna(reference) else ""
    candidate = str(candidate) if pd.notna(candidate) else ""

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {metric: score.fmeasure for metric, score in scores.items()}

def calculate_meteor(reference, candidate):
    """Calculate METEOR score."""
    reference = str(reference) if pd.notna(reference) else ""
    candidate = str(candidate) if pd.notna(candidate) else ""

    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    return meteor_score([reference_tokens], candidate_tokens)

def calculate_bertscore(references, candidates, model_type="bert-base-uncased"):
    """Calculate BERTScore."""
    # Convert to strings and handle NaN
    references = [str(ref) if pd.notna(ref) else "" for ref in references]
    candidates = [str(cand) if pd.notna(cand) else "" for cand in candidates]

    P, R, F1 = score(candidates, references, model_type=model_type, verbose=False)
    return F1.mean().item()

def evaluate_models(df):
    """Evaluate different models based on various metrics."""
    # Identify model columns (all columns except Q and A)
    models = [col for col in df.columns if col not in ["Q", "A"]]
    
    if not models:
        raise ValueError("No model columns found in the DataFrame. Ensure your CSV has columns besides 'Q' and 'A'.")

    metrics = []

    for model in models:
        bleu_scores = []
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        meteor_scores = []

        for _, row in df.iterrows():
            reference = row["A"]
            candidate = row[model]

            # BLEU
            bleu_scores.append(calculate_bleu(reference, candidate))

            # ROUGE
            rouge = calculate_rouge(reference, candidate)
            for key in rouge_scores.keys():
                rouge_scores[key].append(rouge[key])

            # METEOR
            meteor_scores.append(calculate_meteor(reference, candidate))

        # BERTScore
        bertscore = calculate_bertscore(df["A"], df[model])

        # Semantic Similarity
        semantic_similarities = calculate_semantic_similarity(
            df["Q"].fillna("").astype(str),
            df["A"].fillna("").astype(str),
            df[model].fillna("").astype(str)
        )

        # Aggregate scores
        metrics.append({
            "model": model,
            "bleu": sum(bleu_scores) / len(bleu_scores),
            "rouge1": sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"]),
            "rouge2": sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"]),
            "rougeL": sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"]),
            "meteor": sum(meteor_scores) / len(meteor_scores),
            "bertscore": bertscore,
            "semantic_similarity": semantic_similarities
        })

    return pd.DataFrame(metrics)

def main():
    """Main function to run the evaluation."""
    file_path = "qadataset.csv"  # Replace with your CSV file path
    
    try:
        # Load data
        df = load_data(file_path)
        
        # Evaluate models
        results = evaluate_models(df)
        
        # Print and save results
        print("Evaluation Results:")
        print(results)
        
        # Save to CSV
        results.to_csv("evaluation_results.csv", index=False)
        print("\nResults saved to 'evaluation_results.csv'")
    
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    main()