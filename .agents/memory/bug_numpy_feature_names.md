# Bug Pattern: Scikit-Learn RandomForest with NumPy Arrays

## The Issue
When training a `RandomForestClassifier` (or any scikit-learn tree model) using a raw NumPy array (e.g. `np.array(features)`) instead of a Pandas DataFrame, the resulting model object will **not** have the `feature_names_in_` or `feature_names_` attribute.

This causes issues if downstream API endpoints attempt to extract feature names dynamically:
```python
# This condition will fail silently if trained on numpy arrays!
if hasattr(model, "feature_names_"): 
    names = model.feature_names_
```

## The Impact
In the `learn/status` endpoint (`src/api/routers/learn.py`), this bug caused the `feature_importance` array to always return empty `[]`. As a result, the Next.js frontend displayed: "ไม่มีข้อมูล Feature Importance จากโมเดลปัจจุบัน".

## The Fix / Best Practice
When a model is trained using raw NumPy arrays, the features must be mapped manually in the API using a hardcoded array of known feature names, matching the extraction pipeline in the exact same order.

```python
# Correct approach: Hardcode names to match FeatureExtractor order
feature_names = [
    'similarity_score', 'confidence_score',
    'len_p1', 'len_p2', 'len_diff',
    'char_overlap', 'word_overlap',
    # ...
]
if hasattr(learning_system.model.model, "feature_importances_"):
    importances = learning_system.model.model.feature_importances_
    # zip feature_names with importances...
```

**Note:** Always ensure the frontend component (e.g., `page.tsx`) uses the exact same `feature_names` strings as keys for rendering.
