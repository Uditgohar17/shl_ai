
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def load_catalog():
    df = pd.read_csv('attached_assets/shl_catalog.csv')
    df = df.drop_duplicates()
    return df.to_dict('records')

def calculate_duration_score(test_duration, max_duration):
    if not max_duration:
        return 1.0
    test_mins = int(test_duration.split()[0])
    return 1.0 if test_mins <= max_duration else 0.0

def extract_test_types(query):
    test_types = ['Cognitive', 'Behavioral', 'Language']
    return [t for t in test_types if t.lower() in query.lower()]

def extract_max_duration(query):
    import re
    duration_patterns = [
        r'(\d+)\s*min',
        r'(\d+)\s*minutes',
        r'within\s*(\d+)',
        r'less than\s*(\d+)',
        r'under\s*(\d+)'
    ]
    for pattern in duration_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def get_recommendations(query, max_results=10):
    catalog = load_catalog()
    
    # Extract constraints
    max_duration = extract_max_duration(query)
    desired_test_types = extract_test_types(query)
    
    # Prepare documents for TF-IDF
    documents = [f"{item['name']} {item['description']} {item['test_type']}" for item in catalog]
    documents.append(query)
    
    # Calculate TF-IDF similarities
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    
    # Score and rank recommendations
    scored_tests = []
    for idx, item in enumerate(catalog):
        base_score = similarities[idx]
        duration_score = calculate_duration_score(item['duration'], max_duration)
        type_score = 1.2 if not desired_test_types or item['test_type'] in desired_test_types else 0.8
        
        final_score = base_score * duration_score * type_score
        if final_score > 0:
            scored_tests.append((final_score, item))
    
    # Sort and get top recommendations
    scored_tests.sort(reverse=True, key=lambda x: x[0])
    recommendations = []
    seen = set()
    
    for score, item in scored_tests:
        if item['name'] not in seen and len(recommendations) < max_results:
            seen.add(item['name'])
            recommendations.append({
                "name": item['name'],
                "url": item['url'],
                "remote_testing": item['remote'],
                "adaptive_support": item['irt'],
                "duration": item['duration'],
                "test_type": item['test_type'],
                "relevance_score": round(score, 3)
            })
    
    return recommendations if recommendations else [catalog[0]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "API is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
        
    recommendations = get_recommendations(query)
    return jsonify({
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "recommendations": recommendations,
        "metadata": {
            "total_results": len(recommendations),
            "max_duration_constraint": extract_max_duration(query),
            "test_types_requested": extract_test_types(query)
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
