
# Options High-Frequency Trading System

A comprehensive options pricing and trading system integrating multiple machine learning models with real-time data from BigQuery and trading capabilities via Alpaca.

## 🎯 Overview

This system implements the research from "Pricing European Options with Google AutoML, TensorFlow, and XGBoost" by Juan Esteban Berger (University of Notre Dame). It combines six different pricing models into an ensemble system for accurate options pricing and trading signals.

**Research Paper:** https://arxiv.org/abs/2307.00476

## 🏗️ Architecture

### Models Integrated

1. **XGBoost (Max Depth 10)** - Best performing model (MAE: 0.8093) ⭐
2. **Google Cloud AutoML Regressor** (MAE: 1.0248)
3. **XGBoost (Max Depth 5)** (MAE: 1.6362)
4. **5-Layer Feed-Forward Neural Network** (MAE: 4.6374)
5. **3-Layer Feed-Forward Neural Network** (MAE: 8.8075)
6. **Black-Scholes Model** - Baseline (MAE: 8.0082)

### Pre-trained Models

⚠️ **Important:** This system uses **pre-trained models** that must be loaded from file paths. The models are NOT trained during deployment.

**Required Model Files:**
- `xgboost_depth_10.model` - Best model (download from [HuggingFace](https://huggingface.co/juan-esteban-berger/XGBoost_European_Options_Pricing_MD_10))
- `xgboost_depth_5.model`
- `ffnn_5_layer.h5` - 5-Layer Neural Network
- `ffnn_3_layer.h5` - 3-Layer Neural Network

**Model Input Features (27 total):**
1. strike_price
2. implied_volatility
3. zero_coupon_rate
4. index_dividend_yields
5. option_type (1 for call, 0 for put)
6. time_to_maturity
7. underlying_asset_current_price
8-27. lag_1 through lag_20 (past 20 days' closing prices)

📖 **See [docs/MODEL_DEPLOYMENT.md](docs/MODEL_DEPLOYMENT.md) for detailed model deployment instructions.**

### Data Pipeline

**BigQuery Tables:**
- `Security_Name_History` - Stock symbols and names
- `Security_Prices` - Historical stock prices
- `Options_Info` - Options contract details
- `Options_Prices` - Options pricing data
- `Zero_Coupon_Yield_Curve` - Risk-free rates
- `Index_Dividend_Yield` - Dividend yields
- `Exchange_Listing_History` - Exchange information
- `Dividend_Distribution_History` - Dividend payments

### System Components

```
optionsHF/
├── src/
│   ├── connectors/
│   │   └── bigquery.py          # BigQuery data connector (fetches 27 features)
│   ├── models/
│   │   ├── black_scholes.py     # Black-Scholes implementation
│   │   ├── neural_network.py    # Pre-trained FFNN loaders
│   │   ├── xgboost_model.py     # Pre-trained XGBoost loaders
│   │   └── automl.py            # Google AutoML wrapper
│   ├── ensemble.py              # Model ensemble system
│   ├── discord_api.py           # Discord webhook API
│   └── trading_engine.py        # Alpaca trading engine
├── models/                      # Pre-trained model files directory
│   ├── README.md                # Model files documentation
│   ├── xgboost_depth_10.model   # (You need to add this)
│   ├── xgboost_depth_5.model    # (You need to add this)
│   ├── ffnn_5_layer.h5          # (You need to add this)
│   └── ffnn_3_layer.h5          # (You need to add this)
├── docs/
│   └── MODEL_DEPLOYMENT.md      # Detailed deployment guide
├── render-discord.yaml          # Discord service config
├── render-trading.yaml          # Trading service config
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

1. **Pre-trained Model Files** - Download or obtain your trained models
2. **Google Cloud Project** - With BigQuery access
3. **Alpaca Account** - For trading (optional)
4. **Discord Webhook** - For notifications (optional)

### Local Development Setup

1. **Clone Repository:**
   ```bash
   git clone https://github.com/dunavynd2/optionsHF.git
   cd optionsHF
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Model Files:**
   ```bash
   # Create models directory
   mkdir -p models
   
   # Copy your pre-trained model files to models/
   # - xgboost_depth_10.model
   # - xgboost_depth_5.model
   # - ffnn_5_layer.h5
   # - ffnn_3_layer.h5
   ```

5. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials and model paths
   ```

   **Required Variables:**
   ```bash
   # Google Cloud
   GOOGLE_CLOUD_PROJECT=your-project-id
   BIGQUERY_DATASET=options_data
   GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
   
   # Model Paths (Local)
   XGBOOST_DEPTH_10_MODEL_PATH=./models/xgboost_depth_10.model
   XGBOOST_DEPTH_5_MODEL_PATH=./models/xgboost_depth_5.model
   FFNN_5_LAYER_MODEL_PATH=./models/ffnn_5_layer.h5
   FFNN_3_LAYER_MODEL_PATH=./models/ffnn_3_layer.h5
   
   # Optional
   DISCORD_WEBHOOK_URL=your-webhook-url
   ALPACA_API_KEY=your-api-key
   ALPACA_SECRET_KEY=your-secret-key
   ```

6. **Run Services:**
   ```bash
   # Discord API
   python src/discord_api.py
   
   # Trading Engine
   python src/trading_engine.py
   ```

## 📊 Model Performance

Based on research results:

| Model | MAE | MAPE | Status |
|-------|-----|------|--------|
| XGBoost (Depth 10) ⭐ | 0.8093 | 42.23% | Pre-trained |
| AutoML | 1.0248 | 42.73% | Optional |
| XGBoost (Depth 5) | 1.6362 | 187.02% | Pre-trained |
| 5-Layer FFNN | 4.6374 | 243.90% | Pre-trained |
| 3-Layer FFNN | 8.8075 | 323.77% | Pre-trained |
| Black-Scholes | 8.0082 | 63.88% | Built-in |

## 🔧 Render Deployment

### Step 1: Prepare Model Files

Choose one of these methods:

**Method 1: Persistent Disk (Recommended)**
1. Create a persistent disk in Render (1 GB)
2. Mount at `/opt/render/project/src/models`
3. Upload model files via Render Shell

**Method 2: Google Cloud Storage**
1. Upload models to GCS bucket
2. Download in build command

**Method 3: Include in Repository** (only for small models <100 MB)

📖 **See [docs/MODEL_DEPLOYMENT.md](docs/MODEL_DEPLOYMENT.md) for detailed instructions.**

### Step 2: Configure Environment Variables

In Render dashboard, set:

```bash
# Model Paths (Render)
XGBOOST_DEPTH_10_MODEL_PATH=/opt/render/project/src/models/xgboost_depth_10.model
XGBOOST_DEPTH_5_MODEL_PATH=/opt/render/project/src/models/xgboost_depth_5.model
FFNN_5_LAYER_MODEL_PATH=/opt/render/project/src/models/ffnn_5_layer.h5
FFNN_3_LAYER_MODEL_PATH=/opt/render/project/src/models/ffnn_3_layer.h5

# Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
BIGQUERY_DATASET=options_data
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Optional
DISCORD_WEBHOOK_URL=your-webhook-url
ALPACA_API_KEY=your-api-key
ALPACA_SECRET_KEY=your-secret-key
```

### Step 3: Deploy Services

**Discord Service:**
```bash
render deploy --service-type=web --config=render-discord.yaml
```

**Trading Service:**
```bash
render deploy --service-type=worker --config=render-trading.yaml
```

### Step 4: Verify Deployment

Check Render logs for:
```
Successfully loaded pre-trained model from: /opt/render/project/src/models/xgboost_depth_10.model
Successfully loaded pre-trained model from: /opt/render/project/src/models/xgboost_depth_5.model
Successfully loaded pre-trained model from: /opt/render/project/src/models/ffnn_5_layer.h5
Successfully loaded pre-trained model from: /opt/render/project/src/models/ffnn_3_layer.h5
```

## 📡 API Usage

### Discord Webhook API

**Analyze Stock Options:**
```bash
POST /analyze
Content-Type: application/json

{
  "symbol": "AAPL",
  "webhook_url": "https://discord.com/api/webhooks/...",
  "lookback_days": 7
}
```

**Response:**
```json
{
  "success": true,
  "symbol": "AAPL",
  "options_analyzed": 150,
  "analysis": {
    "ensemble_prediction": [245.67],
    "individual_predictions": {
      "XGBoost_Max_Depth_10": [246.12],
      "XGBoost_Max_Depth_5": [244.56],
      "5_Layer_FFNN": [243.21],
      "3_Layer_FFNN": [241.87],
      "Black_Scholes": [247.34]
    },
    "confidence_scores": [0.85],
    "model_load_status": {
      "XGBoost_Max_Depth_10": true,
      "XGBoost_Max_Depth_5": true,
      "5_Layer_FFNN": true,
      "3_Layer_FFNN": true,
      "Black_Scholes": true
    }
  }
}
```

### Trading Engine

```python
from src.trading_engine import TradingEngine

engine = TradingEngine()

# Analyze and trade
result = engine.analyze_and_trade(
    symbol='AAPL',
    quantity=1,
    confidence_threshold=0.7
)
```

## 🧪 Testing

```bash
# Run basic tests
python tests/test_basic.py

# Test model loading
python -c "from src.ensemble import ModelEnsemble; e = ModelEnsemble(); print(e.model_load_status)"
```

## 📚 Documentation

- **[MODEL_DEPLOYMENT.md](docs/MODEL_DEPLOYMENT.md)** - Comprehensive model deployment guide
- **[models/README.md](models/README.md)** - Model files documentation
- **[.env.example](.env.example)** - Environment variables template

## 🔐 Security Notes

- Never commit credentials or model files to Git
- Use environment variables for all secrets
- Enable 2FA on all service accounts
- Use paper trading for testing
- Monitor API usage and costs
- Encrypt sensitive model files

## 🐛 Troubleshooting

### Model Not Loading

**Error:** `FileNotFoundError: Model file not found`

**Solutions:**
1. Verify file exists at specified path
2. Check environment variable is set correctly
3. Ensure file permissions allow reading
4. For Render: Verify persistent disk is mounted

### Missing Features

**Error:** `ValueError: Missing required features`

**Solutions:**
1. Verify BigQuery connector is fetching all 27 features
2. Check lag features are being generated
3. Review feature engineering pipeline

### Prediction Errors

**Error:** `Error predicting with XGBoost_Max_Depth_10`

**Solutions:**
1. Check input data format (must be pandas DataFrame)
2. Verify all 27 features are present
3. Ensure feature order matches model expectations
4. Check for NaN or infinite values

📖 **See [docs/MODEL_DEPLOYMENT.md](docs/MODEL_DEPLOYMENT.md) for more troubleshooting tips.**

## 📚 References

1. **Research Paper:** Berger, J.E. (2023). "Pricing European Options with Google AutoML, TensorFlow, and XGBoost". arXiv:2307.00476
2. **Best Model:** https://huggingface.co/juan-esteban-berger/XGBoost_European_Options_Pricing_MD_10
3. **Alpaca API:** https://alpaca.markets/docs/
4. **Google Cloud BigQuery:** https://cloud.google.com/bigquery/docs

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the repository owner.

## 📄 License

This project is based on research by Juan Esteban Berger, University of Notre Dame.

---

**⚠️ Disclaimer:** This system is for educational and research purposes. Options trading involves significant risk. Always consult with a financial advisor before making investment decisions.

---

**Key Updates in This Version:**
- ✅ Pre-trained model loading (no training during deployment)
- ✅ 27 feature support (including lag features)
- ✅ Comprehensive model deployment documentation
- ✅ Render deployment with persistent disk support
- ✅ Error handling for missing models
- ✅ Model load status tracking
