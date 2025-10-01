
# Pre-trained Model Files

This directory contains the pre-trained machine learning models used by the options pricing system.

## Required Model Files

The system requires the following pre-trained model files:

### XGBoost Models

1. **xgboost_depth_10.model** (Best performing model)
   - MAE: 0.8093
   - Format: XGBoost native format
   - Size: ~50-100 MB (estimated)
   - Download from: [HuggingFace](https://huggingface.co/juan-esteban-berger/XGBoost_European_Options_Pricing_MD_10)

2. **xgboost_depth_5.model**
   - MAE: 1.6362
   - Format: XGBoost native format
   - Size: ~30-60 MB (estimated)

### Neural Network Models

3. **ffnn_5_layer.h5** (5-Layer Feed-Forward Neural Network)
   - MAE: 4.6374
   - Format: Keras/TensorFlow HDF5 format
   - Size: ~10-20 MB (estimated)

4. **ffnn_3_layer.h5** (3-Layer Feed-Forward Neural Network)
   - MAE: 8.8075
   - Format: Keras/TensorFlow HDF5 format
   - Size: ~5-10 MB (estimated)

## Model Input Features

All models expect exactly **27 features** in the following order:

1. `strike_price` - Option strike price
2. `implied_volatility` - Implied volatility
3. `zero_coupon_rate` - Risk-free interest rate
4. `index_dividend_yields` - Dividend yield
5. `option_type` - 1 for call, 0 for put
6. `time_to_maturity` - Time to expiration (years)
7. `underlying_asset_current_price` - Current stock price
8-27. `lag_1` through `lag_20` - Past 20 days' closing prices

## Local Development Setup

1. Create this `models/` directory in your project root:
   ```bash
   mkdir -p models
   ```

2. Download or copy your trained model files to this directory:
   ```bash
   models/
   ├── xgboost_depth_10.model
   ├── xgboost_depth_5.model
   ├── ffnn_5_layer.h5
   └── ffnn_3_layer.h5
   ```

3. Update your `.env` file with local paths:
   ```bash
   XGBOOST_DEPTH_10_MODEL_PATH=./models/xgboost_depth_10.model
   XGBOOST_DEPTH_5_MODEL_PATH=./models/xgboost_depth_5.model
   FFNN_5_LAYER_MODEL_PATH=./models/ffnn_5_layer.h5
   FFNN_3_LAYER_MODEL_PATH=./models/ffnn_3_layer.h5
   ```

## Render Deployment Setup

### Option 1: Using Persistent Disk (Recommended)

1. **Create a Persistent Disk in Render:**
   - Go to your Render dashboard
   - Navigate to your service
   - Click "Disks" → "Add Disk"
   - Name: `model-storage`
   - Size: 1 GB (minimum)
   - Mount Path: `/opt/render/project/src/models`

2. **Upload Model Files:**
   - Use Render Shell to upload files:
     ```bash
     # Connect to Render Shell
     # Then upload files using scp or curl
     cd /opt/render/project/src/models
     curl -o xgboost_depth_10.model <your-download-url>
     ```

3. **Update Environment Variables:**
   ```bash
   XGBOOST_DEPTH_10_MODEL_PATH=/opt/render/project/src/models/xgboost_depth_10.model
   XGBOOST_DEPTH_5_MODEL_PATH=/opt/render/project/src/models/xgboost_depth_5.model
   FFNN_5_LAYER_MODEL_PATH=/opt/render/project/src/models/ffnn_5_layer.h5
   FFNN_3_LAYER_MODEL_PATH=/opt/render/project/src/models/ffnn_3_layer.h5
   ```

### Option 2: Using Cloud Storage (Alternative)

1. **Upload to Google Cloud Storage:**
   ```bash
   gsutil cp xgboost_depth_10.model gs://your-bucket/models/
   gsutil cp xgboost_depth_5.model gs://your-bucket/models/
   gsutil cp ffnn_5_layer.h5 gs://your-bucket/models/
   gsutil cp ffnn_3_layer.h5 gs://your-bucket/models/
   ```

2. **Download in Render Build Command:**
   Add to `render.yaml`:
   ```yaml
   buildCommand: |
     pip install -r requirements.txt
     mkdir -p /opt/render/project/src/models
     gsutil cp gs://your-bucket/models/* /opt/render/project/src/models/
   ```

### Option 3: Include in Repository (Not Recommended for Large Files)

If your model files are small (<100 MB total):

1. Add models to repository:
   ```bash
   git add models/*.model models/*.h5
   git commit -m "Add pre-trained models"
   git push
   ```

2. Update `.gitignore` to allow model files:
   ```
   # Allow model files
   !models/*.model
   !models/*.h5
   ```

3. Use relative paths in `.env`:
   ```bash
   XGBOOST_DEPTH_10_MODEL_PATH=./models/xgboost_depth_10.model
   ```

## Model Loading Code

The system uses the following code to load models:

### XGBoost Models
```python
import xgboost as xgb
model = xgb.Booster()
model.load_model('path-to-your-model-file')
```

### Neural Network Models
```python
from tensorflow import keras
model = keras.models.load_model('path-to-your-model-file')
```

## Troubleshooting

### Model File Not Found
```
FileNotFoundError: Model file not found: /path/to/model
```
**Solution:** Verify the file exists and the path in environment variables is correct.

### Model Loading Error
```
Exception: Failed to load model from /path/to/model
```
**Solution:** 
- Check file format (XGBoost models should be `.model`, neural networks should be `.h5` or `.keras`)
- Verify file is not corrupted
- Ensure compatible library versions (xgboost, tensorflow)

### Missing Features Error
```
ValueError: Missing required features: ['lag_1', 'lag_2', ...]
```
**Solution:** Ensure BigQuery connector is fetching all 27 required features, including lag features.

## Model Performance

Based on research results:

| Model | MAE | MAPE | File Size |
|-------|-----|------|-----------|
| XGBoost Depth 10 | 0.8093 | 42.23% | ~50-100 MB |
| XGBoost Depth 5 | 1.6362 | 187.02% | ~30-60 MB |
| 5-Layer FFNN | 4.6374 | 243.90% | ~10-20 MB |
| 3-Layer FFNN | 8.8075 | 323.77% | ~5-10 MB |

## References

- **Research Paper:** [Pricing European Options with Google AutoML, TensorFlow, and XGBoost](https://arxiv.org/abs/2307.00476)
- **Best Model:** [HuggingFace Repository](https://huggingface.co/juan-esteban-berger/XGBoost_European_Options_Pricing_MD_10)
- **Author:** Juan Esteban Berger, University of Notre Dame

## Security Notes

- **Never commit model files to public repositories** if they contain proprietary training data
- Use environment variables for all file paths
- Consider encrypting model files if they contain sensitive information
- Use Render's persistent disk with appropriate access controls

## Support

For issues with model loading or deployment, please:
1. Check the main README.md for general setup instructions
2. Verify all environment variables are set correctly
3. Ensure model files are in the correct format
4. Check Render logs for detailed error messages

---

**Last Updated:** October 2025
