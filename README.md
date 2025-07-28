# Product Similarity Checker

This project matches new product names against an existing catalog to spot potential duplicates. It loads two CSV files, computes sentence embeddings for each name with the `paraphrase-multilingual-MiniLM-L12-v2` model, and writes the best matches to an output directory.

## Features

- Reads existing products from a CSV containing a `name` column.
- Reads new products from a CSV containing a `รายการ` column and removes duplicate rows.
- Calculates the top three similar old products for every new product.
- Saves results to `matched_products.csv` and logs duplicate entries found in the new list.

## Installation

Use Python 3.8 or newer. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running `main.py`

Run with the default locations (defined in `main.py`):

```bash
python main.py
```

Default paths are:
- `OLD_PRODUCTS_CSV` → `D:/product_checker/cleaned_products.csv`
- `NEW_PRODUCTS_CSV` → `D:/bill26668/_ocr_output/merged_receipts.csv`
- `OUTPUT_DIR` → `D:/product_checker`

### Supplying Custom Paths

You can override the defaults by setting environment variables before running:

```bash
export OLD_PRODUCTS_CSV=/path/to/old_products.csv
export NEW_PRODUCTS_CSV=/path/to/new_products.csv
export OUTPUT_DIR=/path/to/output
python main.py
```

Alternatively, call the `run` function directly from the command line with your own paths:

```bash
python -c "import main; main.run('old_products.csv', 'new_products.csv', 'output_dir')"
```

The output directory will contain `matched_products.csv` with similarity scores and `duplicate_new_products.csv` if duplicate new entries were detected.
