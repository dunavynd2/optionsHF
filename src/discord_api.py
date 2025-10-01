
"""
Discord Webhook API
Flask API for sending stock analysis to Discord
"""

import os
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import traceback

from src.connectors.bigquery import BigQueryConnector
from src.ensemble import ModelEnsemble


app = Flask(__name__)

# Initialize components
bq_connector = None
ensemble = None


def initialize_services():
    """Initialize BigQuery and model ensemble"""
    global bq_connector, ensemble
    
    try:
        bq_connector = BigQueryConnector()
        ensemble = ModelEnsemble(use_automl=False)  # AutoML disabled by default
        print("Services initialized successfully")
    except Exception as e:
        print(f"Error initializing services: {e}")
        traceback.print_exc()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'discord-webhook-api'
    })


@app.route('/analyze', methods=['POST'])
def analyze_stock():
    """
    Analyze stock options and send to Discord
    
    Request body:
    {
        "symbol": "AAPL",
        "webhook_url": "https://discord.com/api/webhooks/...",
        "lookback_days": 7
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        symbol = data.get('symbol')
        webhook_url = data.get('webhook_url') or os.getenv('DISCORD_WEBHOOK_URL')
        lookback_days = data.get('lookback_days', 7)
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        if not webhook_url:
            return jsonify({'error': 'Discord webhook URL is required'}), 400
        
        # Fetch options data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        options_df = bq_connector.get_options_data(symbol, start_date, end_date)
        
        if options_df.empty:
            return jsonify({
                'error': f'No options data found for {symbol}',
                'symbol': symbol,
                'date_range': f'{start_date.date()} to {end_date.date()}'
            }), 404
        
        # Prepare features
        features_df = bq_connector.prepare_features(options_df)
        
        # Analyze with ensemble
        analysis = ensemble.analyze_option(features_df)
        
        # Send to Discord
        import requests
        
        # Format message
        message = format_discord_message(symbol, analysis, features_df)
        
        discord_response = requests.post(
            webhook_url,
            json={'content': message}
        )
        
        if discord_response.status_code != 204:
            return jsonify({
                'error': 'Failed to send to Discord',
                'discord_status': discord_response.status_code
            }), 500
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'options_analyzed': len(features_df),
            'analysis': analysis,
            'discord_sent': True
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


def format_discord_message(symbol: str, analysis: dict, features_df) -> str:
    """Format analysis results for Discord"""
    
    # Get first option details
    first_option = features_df.iloc[0]
    
    message = f"""
**Options Analysis for {symbol}**
📊 **Option Details:**
- Strike: ${first_option['strike_price']:.2f}
- Underlying: ${first_option['underlying_price']:.2f}
- Days to Expiration: {first_option['days_to_expiration']:.0f}
- Type: {'Call' if first_option.get('is_call', 1) else 'Put'}

💰 **Ensemble Prediction:** ${analysis['ensemble_prediction'][0]:.2f}

📈 **Individual Model Predictions:**
- XGBoost (Depth 10): ${analysis['individual_predictions']['XGBoost_Max_Depth_10'][0]:.2f}
- XGBoost (Depth 5): ${analysis['individual_predictions']['XGBoost_Max_Depth_5'][0]:.2f}
- 5-Layer FFNN: ${analysis['individual_predictions']['5_Layer_FFNN'][0]:.2f}
- 3-Layer FFNN: ${analysis['individual_predictions']['3_Layer_FFNN'][0]:.2f}
- Black-Scholes: ${analysis['individual_predictions']['Black_Scholes'][0]:.2f}

🎯 **Confidence Score:** {analysis['confidence_scores'][0]:.2%}

📊 **Statistics:**
- Mean: ${analysis['statistics']['mean'][0]:.2f}
- Median: ${analysis['statistics']['median'][0]:.2f}
- Std Dev: ${analysis['statistics']['std'][0]:.2f}

⏰ Analysis Time: {analysis['timestamp']}
"""
    
    return message


if __name__ == '__main__':
    initialize_services()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
