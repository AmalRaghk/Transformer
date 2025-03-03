from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from transformer import SimplifiedTransformer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for all routes

# Initialize the transformer model
transformer = SimplifiedTransformer(d_model=64, nhead=4, dim_feedforward=128)
@app.route('/api/process', methods=['POST'])
def process_input():
    try:
        data = request.json
        if not data or 'input' not in data:
            raise ValueError("Invalid input: 'input' key missing or empty")
        
        input_text = data.get('input', '')

        # Simple tokenization (split by space)
        tokens = input_text.split()

        # Get visualization data
        vis_data = transformer.get_visualization_data(tokens)

        return jsonify(vis_data)
    except Exception as e:
        app.logger.error(f"Error in process_input: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
