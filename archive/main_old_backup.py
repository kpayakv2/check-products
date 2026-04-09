import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from sentence_transformers import SentenceTransformer, util
import torch


def prompt_csv_path(prompt_text: str) -> Path:
    """Interactively prompt for a CSV file path until it exists."""
    while True:
        available = [p.name for p in Path(".").glob("*.csv")]
        if available:
            print("Available CSV files: " + ", ".join(available))
        path_str = input(prompt_text).strip()
        path = Path(path_str.strip("'\"")).expanduser().resolve()
        if path.is_file():
            return path
        print(f"{path} is not a valid file. Please try again without quotes.")


def check_product_similarity(
    new_product: str,
    old_product_names: List[str],
    old_embeddings,
    model: SentenceTransformer,
    top_k: int = 3,
    new_embedding: Optional[torch.Tensor] = None,
) -> List[Tuple[str, float]]:
    """
    Compute the similarity between a new product name and a list of old product names.
    Returns the top_k most similar old product names with their cosine similarity scores.

    Parameters:
        new_product (str): The name of the new product to compare.
        old_product_names (List[str]): List of old product names corresponding to old_embeddings.
        old_embeddings: Embeddings for old_product_names (as a PyTorch tensor or NumPy array).
        model (SentenceTransformer):
            The sentence transformer model to use for encoding the new product.
        top_k (int): The number of top similar results to return (default 3).
        new_embedding (Optional[torch.Tensor]): Pre-computed embedding for new_product.
            If None, will be computed using the model.

    Returns:
        List[Tuple[str, float]]: A list of tuples (old_product_name, similarity_score) for the
            top_k similar old products.
    """
    # Verify that each product name has a corresponding embedding
    embedding_count = (
        old_embeddings.shape[0] if hasattr(old_embeddings, "shape") else len(old_embeddings)
    )
    if len(old_product_names) != embedding_count:
        raise ValueError("old_embeddings first dimension must match length of old_product_names")

    # Encode the new product name into the same embedding space as old_product_names
    if new_embedding is None:
        new_embedding = model.encode([new_product], convert_to_tensor=True)
    else:
        # Ensure the embedding is a 2D tensor
        if new_embedding.dim() == 1:
            new_embedding = new_embedding.unsqueeze(0)

    # Validate tensor shapes before computing similarity
    print(f"Debug - new_embedding shape: {new_embedding.shape}")
    print(f"Debug - old_embeddings shape: {old_embeddings.shape}")
    
    # Ensure both tensors have compatible shapes
    if new_embedding.shape[-1] != old_embeddings.shape[-1]:
        raise ValueError(
            f"Embedding dimension mismatch: new_embedding.shape[-1]={new_embedding.shape[-1]}, "
            f"old_embeddings.shape[-1]={old_embeddings.shape[-1]}"
        )

    # Compute cosine similarity between the new product embedding and all old product embeddings
    cos_scores = util.cos_sim(new_embedding, old_embeddings)[0]

    # Determine the number of results to return without exceeding available products
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    effective_k = min(top_k, len(old_product_names))

    # Get the highest similarity scores and their indices
    top_results = cos_scores.topk(k=effective_k)

    # Pair each top score with the corresponding old product name
    return [
        (old_product_names[int(idx)], score.item())
        for score, idx in zip(top_results[0], top_results[1])
    ]


def remove_duplicates(
    df: pd.DataFrame, subset: str = "รายการ", duplicates_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicate rows from a DataFrame based on a key column or columns.
    If duplicates are found and duplicates_path is provided, save all duplicate rows to a CSV file.
    Parameters:
        df (pd.DataFrame): The DataFrame to check for duplicates.
        subset (str):
            Column name (or list of column names) to consider for finding duplicates.
            Defaults to 'รายการ'.
        duplicates_path (Optional[Path]):
            Path to save duplicate entries CSV if duplicates are found.
            If None, duplicates are not saved.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - The DataFrame after removing duplicate rows
              (keeping the first occurrence of each duplicate).
            - A DataFrame of duplicate rows that were found (empty if no duplicates).
    """
    # Identify all rows that are duplicated in the specified subset
    # (including all occurrences of the duplicates)
    duplicates_df = df[df.duplicated(subset=[subset], keep=False)]
    # If any duplicates found and an output path is provided, save them
    if not duplicates_df.empty and duplicates_path is not None:
        duplicates_df.to_csv(duplicates_path, index=False, encoding="utf-8-sig")
    # Remove duplicate rows (keeping the first occurrence of each duplicate)
    deduped_df = df.drop_duplicates(subset=[subset])
    return deduped_df, duplicates_df


def run(
    old_products_csv: Optional[Path] = None,
    new_products_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Run the product name comparison process:
     1. Load old products and new products from CSV files.
     2. Remove duplicate product names from the new products list.
     3. Use a SentenceTransformer model to find top 3 similar old products for each new product.
     4. Save the top matches for each new product to an output CSV file.
    Environment Variables:
        OLD_PRODUCTS_CSV:
            Override path to the old products CSV file (should contain a column 'name').
        NEW_PRODUCTS_CSV:
            Override path to the new products CSV file (should contain a column 'รายการ').
        OUTPUT_DIR: Override directory for output files.
    Parameters:
        old_products_csv (Optional[Path]):
            Path to the CSV file of old products. If None, uses OLD_PRODUCTS_CSV env var
            or default path.
        new_products_csv (Optional[Path]):
            Path to the CSV file of new products. If None, uses NEW_PRODUCTS_CSV env var
            or default path.
        output_dir (Optional[Path]):
            Directory for output files. If None, uses OUTPUT_DIR env var or default path.
    Returns:
        None. This function writes output files and prints status messages.
    """
    # Determine file paths using parameters, environment variables, or defaults
    old_products_csv = (
        (
            Path(os.getenv("OLD_PRODUCTS_CSV", "old_products.csv"))
            if old_products_csv is None
            else Path(old_products_csv)
        )
        .expanduser()
        .resolve()
    )
    new_products_csv = (
        (
            Path(os.getenv("NEW_PRODUCTS_CSV", "new_products.csv"))
            if new_products_csv is None
            else Path(new_products_csv)
        )
        .expanduser()
        .resolve()
    )
    output_dir = (
        (Path(os.getenv("OUTPUT_DIR", "output")) if output_dir is None else Path(output_dir))
        .expanduser()
        .resolve()
    )
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load datasets
    old_products_df = pd.read_csv(old_products_csv)
    new_products_df = pd.read_csv(new_products_csv)
    # Extract product name lists
    old_product_names: List[str] = old_products_df["name"].tolist()
    # Remove duplicate new product names
    duplicates_path = output_dir / "duplicate_new_products.csv"
    new_products_df_deduped, duplicates_df = remove_duplicates(
        new_products_df, subset="รายการ", duplicates_path=duplicates_path
    )
    # If duplicates were found, inform the user
    if not duplicates_df.empty:
        # Print number of duplicate entries (counting all occurrences of duplicates)
        print(f"พบสินค้าซ้ำ {len(duplicates_df)} รายการ บันทึกที่ {duplicates_path}")
    # Update new product names list after removing duplicates
    new_product_names: List[str] = new_products_df_deduped["รายการ"].tolist()
    # Load pre-trained model for multilingual sentence similarity
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # Compute embeddings for all old product names (for efficiency, do this once)
    old_embeddings = model.encode(old_product_names, convert_to_tensor=True)
    name_to_index = {name: idx for idx, name in enumerate(old_product_names)}
    # Prepare a list to collect the similarity results
    output_rows: List[dict] = []
    # Compute top 3 similar old products for each new product
    for new_product in new_product_names:
        try:
            new_vec = model.encode([new_product], convert_to_tensor=True)
            top_matches = check_product_similarity(
                new_product, old_product_names, old_embeddings, model, top_k=3, new_embedding=new_vec
            )
            for old_name, score in top_matches:
                row = {
                    "new_product": new_product,
                    "new_product_vector": new_vec[0].tolist(),
                    "matched_old_product": old_name,
                    "score": score,
                }
                old_idx = name_to_index.get(old_name)
                if old_idx is not None:
                    row["matched_old_vector"] = old_embeddings[old_idx].tolist()
                output_rows.append(row)
        except Exception as e:
            print(f"Error processing product '{new_product}': {e}")
            print(f"  new_vec shape: {new_vec.shape if 'new_vec' in locals() else 'N/A'}")
            print(f"  old_embeddings shape: {old_embeddings.shape}")
            print(f"  Skipping this product and continuing...")
            continue  # ข้ามไป product ถัดไปแทนที่จะ crash
    # Save the matching results to CSV
    output_df = pd.DataFrame(output_rows)
    matched_path = output_dir / "matched_products.csv"
    output_df.to_csv(matched_path, index=False, encoding="utf-8-sig")
    print(
        f"บันทึกผลลัพธ์ที่ {matched_path}",
        "เรียบร้อยแล้ว",
        "(encoding utf-8-sig สำหรับ Excel)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match new products to existing catalog")
    parser.add_argument("--old-products-csv", help="CSV of existing products")
    parser.add_argument("--new-products-csv", help="CSV of new products to check")
    parser.add_argument("--output-dir", help="Directory for output files")
    args = parser.parse_args()

    old_csv = args.old_products_csv or os.getenv("OLD_PRODUCTS_CSV")
    new_csv = args.new_products_csv or os.getenv("NEW_PRODUCTS_CSV")

    if old_csv is None:
        old_csv = prompt_csv_path("Path to old products CSV: ")
    if new_csv is None:
        new_csv = prompt_csv_path("Path to new products CSV: ")

    run(old_csv, new_csv, args.output_dir)
