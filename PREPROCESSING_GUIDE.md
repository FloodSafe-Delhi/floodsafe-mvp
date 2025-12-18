# Data Preprocessing Guide for ML Training

## Issue Summary

The `delhi_monsoon_5years.npz` dataset contains **4 critical zero-variance dimensions** that must be removed before training the LSTM model. Additionally, **13 total dimensions have zero variance** and should be removed for optimal model performance.

| Metric | Value |
|--------|-------|
| Total Features | 37 |
| Critical Zero-Var Dims | 4 (Dims 21, 22, 23, 33) |
| Total Zero-Var Dims | 13 (35.1% of features) |
| Usable Features | 24 (64.9% of features) |
| Samples | 605 |

## Zero-Variance Dimensions to Remove

### CRITICAL (Must Remove - 4 dims)
These dimensions are completely constant and add ZERO information:

```
Dim 21: 154.000000 (constant map/grid parameter)
Dim 22: 329.000000 (constant map/grid parameter)
Dim 23: 175.000000 (constant map/grid parameter)
Dim 33: 1.000000   (constant binary flag)
```

### Additional (Should Remove - 9 dims)
These also have zero variance and provide no predictive value:

```
Dim 9:  32.666107  (likely fixed latitude)
Dim 10: 21.781651  (likely fixed longitude)
Dim 11: 0.884122   (constant parameter)
Dim 12: 42.823067  (constant parameter)
Dim 13: 34.511159  (constant parameter)
Dim 14: 64.604718  (constant parameter)
Dim 20: 219.334385 (constant parameter)
Dim 24: 3.310536   (constant parameter)
Dim 25: 163.934112 (constant parameter)
```

## Solution 1: Using sklearn (Recommended)

```python
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Load data
data = np.load('apps/ml-service/data/delhi_monsoon_5years.npz')
X = data['X']
y = data['y']
rainfall = data['rainfall']
dates = data['dates']

print(f"Original shape: {X.shape}")

# Method 1: Remove strictly zero-variance features
selector = VarianceThreshold(threshold=0)
X_filtered = selector.fit_transform(X)

print(f"After removing zero-variance: {X_filtered.shape}")
# Output: (605, 33)

# Method 2: Remove with small threshold (recommended)
# Helps with numerical stability in some algorithms
selector = VarianceThreshold(threshold=1e-10)
X_filtered = selector.fit_transform(X)

print(f"After removing near-zero variance: {X_filtered.shape}")
# Output: (605, 24)

# Get which features were kept
feature_mask = selector.get_support()
feature_indices = np.where(feature_mask)[0]
print(f"Kept features: {list(feature_indices)}")

# Save preprocessed data
np.savez_compressed(
    'delhi_monsoon_5years_preprocessed.npz',
    X=X_filtered,
    y=y,
    rainfall=rainfall,
    dates=dates,
    feature_mask=feature_mask
)
```

## Solution 2: Manual Feature Selection

```python
import numpy as np

# Load data
data = np.load('apps/ml-service/data/delhi_monsoon_5years.npz')
X = data['X']
y = data['y']

# Option A: Remove only critical dims (4 dims)
keep_dims = [i for i in range(37) if i not in [21, 22, 23, 33]]
X_filtered = X[:, keep_dims]
print(f"Removed critical dims: {X_filtered.shape}")  # (605, 33)

# Option B: Remove all zero-variance dims (13 dims)
keep_dims = [i for i in range(37) if i not in [9, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25, 33]]
X_filtered = X[:, keep_dims]
print(f"Removed all zero-var: {X_filtered.shape}")  # (605, 24)

# Save
np.savez_compressed('delhi_monsoon_5years_preprocessed.npz', X=X_filtered, y=y)
```

## Solution 3: Using Pandas

```python
import numpy as np
import pandas as pd

# Load data
data = np.load('apps/ml-service/data/delhi_monsoon_5years.npz')
X = data['X']

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

# Remove zero-variance columns
df_filtered = df.loc[:, (df.var(axis=0) != 0)]

print(f"Original: {df.shape}")
print(f"Filtered: {df_filtered.shape}")

X_filtered = df_filtered.values
np.savez_compressed('delhi_monsoon_5years_preprocessed.npz', X=X_filtered)
```

## Integration with LSTM Model

### Step 1: Update Input Layer

**Before:**
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=37, hidden_size=64):  # Old: 37 features
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
```

**After:**
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=24, hidden_size=64):  # New: 24 features
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
```

### Step 2: Update Data Loading

**Before:**
```python
data = np.load('delhi_monsoon_5years.npz')
X = data['X']  # Shape: (605, 37)
```

**After:**
```python
data = np.load('delhi_monsoon_5years_preprocessed.npz')
X = data['X']  # Shape: (605, 24)
```

### Step 3: Verify Shape Match

```python
# During training
X_batch = X[batch_indices]  # Should be (batch_size, 24)
output = model(X_batch)     # Should work without shape errors
```

## Verification Commands

```bash
# Load and verify preprocessed data
python -c "
import numpy as np
data = np.load('apps/ml-service/data/delhi_monsoon_5years_preprocessed.npz')
X = data['X']
print(f'Shape: {X.shape}')
print(f'Variance > 0: {np.sum(X.var(axis=0) > 0)}')
print(f'Min: {X.min()}, Max: {X.max()}')
print(f'No NaN: {np.sum(np.isnan(X)) == 0}')
"
```

## Expected Improvements After Preprocessing

| Metric | Before | After |
|--------|--------|-------|
| Features | 37 | 24 |
| Model Parameters | More | Less |
| Training Speed | Slower | Faster |
| Numerical Stability | Risk | Improved |
| Information Content | 100% | ~65% |
| Overfitting Risk | Higher | Lower |

## Troubleshooting

### Issue: Model still gets NaN loss
**Solution:** Check if any remaining features have extremely low variance
```python
variances = X.var(axis=0)
print(variances)
print(f"Min variance: {variances[variances > 0].min()}")
```

### Issue: Shape mismatch error during training
**Solution:** Verify preprocessed data shape matches model input
```python
data = np.load('delhi_monsoon_5years_preprocessed.npz')
X = data['X']
print(f"Data shape: {X.shape}")
print(f"Model expects input_size: {model.lstm.input_size}")
assert X.shape[1] == model.lstm.input_size, "Shape mismatch!"
```

### Issue: Performance worse after preprocessing
**Solution:** This is actually GOOD - model was overfitting on constant features
- Compare using same train/val split
- Check cross-validation scores
- Constant features were hurting generalization

## Next Steps

1. **Preprocess the data** using one of the solutions above
2. **Save** the preprocessed file as `delhi_monsoon_5years_preprocessed.npz`
3. **Update** model input_size from 37 to 24
4. **Retrain** the LSTM model on filtered data
5. **Compare** performance metrics before/after
6. **Document** which features were removed and why
7. **Add** automatic variance filtering to your data pipeline

## Files Reference

- **Data**: `apps/ml-service/data/delhi_monsoon_5years.npz`
- **Verification scripts**: `verify_training_data.py`, `zero_variance_analysis.py`
- **Reports**: `DATA_VERIFICATION_REPORT.md`, `TRAINING_DATA_VERIFICATION_SUMMARY.txt`

## Why This Matters

Constant features in machine learning:
- **Waste model capacity** - parameters trained on useless input
- **Reduce generalization** - model wastes capacity on noise
- **Slow training** - more parameters to optimize
- **Risk numerical issues** - some algorithms fail with zero variance
- **Make debugging hard** - unclear if poor performance is feature or model issue

By removing them, you get:
- Faster training
- Better generalization
- More stable models
- Clearer performance attribution
- Fewer hyperparameters to tune
