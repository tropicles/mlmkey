from flask import Flask, request, jsonify
from predict import KeywordExtractor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
extractor = KeywordExtractor()

@app.route('/extract-keywords', methods=['POST'])
def api_extract_keywords():
    try:
        data = request.get_json()
        if not data or 'job_description' not in data:
            return jsonify({"error": "Invalid request format. 'job_description' key is required."}), 400

        num_keywords = data.get('num_keywords', 20)
        keywords = extractor.extract_keywords(data['job_description'], num_keywords=num_keywords)
        
        return jsonify({
            "keywords": keywords,
            "count": len(keywords),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)