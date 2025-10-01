
# Options High-Frequency Trading System

A comprehensive options pricing and trading system integrating multiple machine learning models with real-time data from BigQuery and trading capabilities via Alpaca.

## 🎯 Overview

This system implements the research from "Pricing European Options with Google AutoML, TensorFlow, and XGBoost" by Juan Esteban Berger (University of Notre Dame). It combines six different pricing models into an ensemble system for accurate options pricing and trading signals.

**Research Paper:** https://arxiv.org/abs/2307.00476

## 🏗️ Architecture

### Models Integrated

1. **XGBoost (Max Depth 10)** - Best performing model (MAE: 0.8093)
2. **Google Cloud AutoML Regressor** (MAE: 1.0248)
3. **XGBoost (Max Depth 5)** (MAE: 1.6362)
4. **5-Layer Feed-Forward Neural Network** (MAE: 4.6374)
5. **3-Layer Feed-Forward Neural Network** (MAE: 8.8075)
6. **Black-Scholes Model** - Baseline (MAE: 8.0082)

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
│   │   └── bigquery.py          # BigQuery data connector
│   ├── models/
│   │   ├── black_scholes.py     # Black-Scholes implementation
│   │   ├── neural_network.py    # 3-Layer & 5-Layer FFNN
│   │   ├── xgboost_model.py     # XGBoost models
│   │   └── automl.py            # Google AutoML wrapper
│   ├── ensemble.py              # Model ensemble system
│   ├── discord_api.py           # Discord webhook API
│   └── trading_engine.py        # Alpaca trading engine
├── render-discord.yaml          # Discord service config
├── render-trading.yaml          # Trading service config
└── requirements.txt             # Python dependencies
```

## 🚀 Deployment

### Two Separate Render Services

#### Service 1: Discord Webhook API
```yaml
# render-discord.yaml
- Flask API for stock analysis
- Endpoints: /health, /analyze
- Sends analysis to Discord webhooks
```

**Environment Variables:**
- `DISCORD_WEBHOOK_URL` - Discord webhook URL
- `GOOGLE_CLOUD_PROJECT` - GCP project ID
- `BIGQUERY_DATASET` - BigQuery dataset name
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP credentials JSON

#### Service 2: Alpaca Trading & Backtesting
```yaml
# render-trading.yaml
- Background worker for trading
- Alpaca integration
- Backtesting engine
```

**Environment Variables:**
- `ALPACA_API_KEY` - Alpaca API key
- `ALPACA_SECRET_KEY` - Alpaca secret key
- `ALPACA_BASE_URL` - Alpaca API URL (paper/live)
- `GOOGLE_CLOUD_PROJECT` - GCP project ID
- `BIGQUERY_DATASET` - BigQuery dataset name
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP credentials JSON

## 📊 Model Performance

Based on research results:

| Model | MAE | MAPE | Training Time |
|-------|-----|------|---------------|
| XGBoost (Depth 10) | 0.8093 | 42.23% | 1917s |
| AutoML | 1.0248 | 42.73% | 174420s |
| XGBoost (Depth 5) | 1.6362 | 187.02% | 971s |
| 5-Layer FFNN | 4.6374 | 243.90% | 3288s |
| 3-Layer FFNN | 8.8075 | 323.77% | 3066s |
| Black-Scholes | 8.0082 | 63.88% | N/A |

## 🔧 Setup Instructions

### 1. Google Cloud Setup

```bash
# Create BigQuery dataset
bq mk --dataset --location=US options_data

# Load data into BigQuery tables
bq load --source_format=CSV options_data.Security_Prices gs://your-bucket/security_prices.csv
bq load --source_format=CSV options_data.Options_Prices gs://your-bucket/options_prices.csv
# ... (load other tables)

# Create service account
gcloud iam service-accounts create optionshf-service \
    --display-name="OptionsHF Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:optionshf-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer"

# Download credentials
gcloud iam service-accounts keys create credentials.json \
    --iam-account=optionshf-service@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 2. Alpaca Setup

1. Sign up at https://alpaca.markets
2. Get API keys from dashboard
3. Use paper trading URL for testing: `https://paper-api.alpaca.markets`

### 3. Discord Setup

1. Create Discord webhook in your server
2. Copy webhook URL
3. Add to environment variables

### 4. Deploy to Render

**Discord Service:**
```bash
# Deploy from GitHub
render deploy --service-type=web --config=render-discord.yaml
```

**Trading Service:**
```bash
# Deploy from GitHub
render deploy --service-type=worker --config=render-trading.yaml
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
      "AutoML_Regressor": [245.89],
      "XGBoost_Max_Depth_5": [244.56],
      "5_Layer_FFNN": [243.21],
      "3_Layer_FFNN": [241.87],
      "Black_Scholes": [247.34]
    },
    "confidence_scores": [0.85],
    "statistics": {
      "mean": [245.50],
      "median": [245.67],
      "std": [1.89]
    }
  },
  "discord_sent": true
}
```

### Trading Engine

**Analyze and Trade:**
```python
from src.trading_engine import TradingEngine

engine = TradingEngine()

# Analyze option
result = engine.analyze_and_trade(
    symbol='AAPL',
    quantity=1,
    confidence_threshold=0.7
)

print(result)
# {
#   'symbol': 'AAPL',
#   'prediction': 245.67,
#   'confidence': 0.85,
#   'trade_signal': 'BUY',
#   'message': 'High confidence (85%) - Trade signal generated'
# }
```

**Backtest Strategy:**
```python
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

results = engine.backtest(
    symbol='AAPL',
    start_date=start_date,
    end_date=end_date,
    initial_capital=10000.0
)

print(results)
```

## 🧪 Local Development

```bash
# Clone repository
git clone https://github.com/dunavynd2/optionsHF.git
cd optionsHF

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
export BIGQUERY_DATASET="options_data"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
export DISCORD_WEBHOOK_URL="your-webhook-url"
export ALPACA_API_KEY="your-api-key"
export ALPACA_SECRET_KEY="your-secret-key"

# Run Discord API
python src/discord_api.py

# Run trading engine
python src/trading_engine.py
```

## 🔬 Model Training

To train models with your own data:

```python
from src.models.xgboost_model import XGBoostMaxDepth10
from src.models.neural_network import FiveLayerFFNN
import pandas as pd

# Load training data
train_df = pd.read_csv('training_data.csv')
X_train = train_df[feature_columns].values
y_train = train_df['option_price'].values

# Train XGBoost
xgb_model = XGBoostMaxDepth10()
xgb_model.train(X_train, y_train)
xgb_model.save('models/xgboost_depth10.joblib')

# Train Neural Network
nn_model = FiveLayerFFNN()
nn_model.train(X_train, y_train, epochs=100)
nn_model.save('models/5layer_ffnn.h5')
```

## 📈 Feature Engineering

The system uses the following features for prediction:

**Core Features:**
- `underlying_price` - Current stock price
- `strike_price` - Option strike price
- `time_to_expiration` - Time to expiration (years)
- `risk_free_rate` - Risk-free interest rate
- `dividend_yield` - Dividend yield
- `moneyness` - S/K ratio
- `is_call` - Option type (1=call, 0=put)

**Additional Features (XGBoost):**
- `volume` - Trading volume
- `open_interest` - Open interest
- `relative_spread` - Bid-ask spread / mid price

## 🎯 Ensemble Strategy

The ensemble combines predictions using weighted averaging based on model performance:

```python
# Weights based on inverse MAE
weights = {
    'XGBoost_Max_Depth_10': 0.8093,    # Highest weight
    'AutoML_Regressor': 1.0248,
    'XGBoost_Max_Depth_5': 1.6362,
    '5_Layer_FFNN': 4.6374,
    '3_Layer_FFNN': 8.8075,
    'Black_Scholes': 8.0082
}

# Normalized weights sum to 1.0
ensemble_prediction = Σ(weight_i × prediction_i) / Σ(weight_i)
```

**Confidence Scoring:**
- Based on model agreement (coefficient of variation)
- Higher agreement = higher confidence
- Range: 0.0 to 1.0

## 🔐 Security Notes

- Never commit credentials to Git
- Use environment variables for all secrets
- Enable 2FA on all service accounts
- Use paper trading for testing
- Monitor API usage and costs

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
