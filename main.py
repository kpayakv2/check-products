import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer, util


def check_product_similarity(
    new_product: str,
    old_product_names: List[str],
    old_embeddings,
    model: SentenceTransformer,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Compute the similarity between a new product name and a list of old product names using embeddings.
    Returns the top_k most similar old product names with their cosine similarity scores.
    Parameters:
        new_product (str): The name of the new product to compare.
        old_product_names (List[str]): List of old product names corresponding to old_embeddings.
        old_embeddings: Embeddings for old_product_names (as a PyTorch tensor or NumPy array).
        model (SentenceTransformer): The sentence transformer model to use for encoding the new product.
        top_k (int): The number of top similar results to return (default 3).
    Returns:
        List[Tuple[str, float]]: A list of tuples (old_product_name, similarity_score) for the top_k similar old products.
    """
    # Encode the new product name into the same embedding space as old_product_names
    new_embedding = model.encode([new_product], convert_to_tensor=True)
    # Compute cosine similarity between the new product embedding and all old product embeddings
    cos_scores = util.cos_sim(new_embedding, old_embeddings)[0]
    # Get the top_k highest similarity scores and their indices
    top_results = cos_scores.topk(k=top_k)
    result: List[Tuple[str, float]] = []
    # Pair each top score with the corresponding old product name
    for score, idx in zip(top_results[0], top_results[1]):
        result.append((old_product_names[int(idx)], float(score)))
    return result


def remove_duplicates(
    df: pd.DataFrame, subset: str = "รายการ", duplicates_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicate rows from a DataFrame based on a key column or columns.
    If duplicates are found and duplicates_path is provided, save all duplicate rows to a CSV file.
    Parameters:
        df (pd.DataFrame): The DataFrame to check for duplicates.
        subset (str): Column name (or list of column names) to consider for finding duplicates. Defaults to 'รายการ'.
        duplicates_path (Optional[Path]): Path to save duplicate entries CSV if duplicates are found. If None, duplicates are not saved.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - The DataFrame after removing duplicate rows (keeping the first occurrence of each duplicate).
            - A DataFrame of duplicate rows that were found (empty if no duplicates).
    """
    # Identify all rows that are duplicated in the specified subset (including all occurrences of the duplicates)
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
        OLD_PRODUCTS_CSV: Override path to the old products CSV file (should contain a column 'name').
        NEW_PRODUCTS_CSV: Override path to the new products CSV file (should contain a column 'รายการ').
        OUTPUT_DIR: Override directory for output files.
    Parameters:
        old_products_csv (Optional[Path]): Path to the CSV file of old products. If None, uses OLD_PRODUCTS_CSV env var or default path.
        new_products_csv (Optional[Path]): Path to the CSV file of new products. If None, uses NEW_PRODUCTS_CSV env var or default path.
        output_dir (Optional[Path]): Directory for output files. If None, uses OUTPUT_DIR env var or default path.
    Returns:
        None. This function writes output files and prints status messages.
    """
    # Determine file paths using parameters, environment variables, or defaults
    old_products_csv = (
        Path(os.getenv("OLD_PRODUCTS_CSV", "old_products.csv"))
        if old_products_csv is None
        else Path(old_products_csv)
    )
    new_products_csv = (
        Path(os.getenv("NEW_PRODUCTS_CSV", "new_products.csv"))
        if new_products_csv is None
        else Path(new_products_csv)
    )
    output_dir = Path(os.getenv("OUTPUT_DIR", "output")) if output_dir is None else Path(output_dir)
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
    # Prepare a list to collect the similarity results
    output_rows: List[dict] = []
    # Compute top 3 similar old products for each new product
    for new_product in new_product_names:
        top_matches = check_product_similarity(
            new_product, old_product_names, old_embeddings, model, top_k=3
        )
        for old_name, score in top_matches:
            output_rows.append(
                {"new_product": new_product, "matched_old_product": old_name, "score": score}
            )
    # Save the matching results to CSV
    output_df = pd.DataFrame(output_rows)
    matched_path = output_dir / "matched_products.csv"
    output_df.to_csv(matched_path, index=False, encoding="utf-8-sig")
    print(f"บันทึกผลลัพธ์ที่ {matched_path} เรียบร้อยแล้ว (encoding utf-8-sig สำหรับ Excel)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match new products to existing catalog")
    parser.add_argument("--old-products-csv", help="CSV of existing products")
    parser.add_argument("--new-products-csv", help="CSV of new products to check")
    parser.add_argument("--output-dir", help="Directory for output files")
    args = parser.parse_args()
    run(args.old_products_csv, args.new_products_csv, args.output_dir)
