from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import re
import json
from collections import defaultdict, Counter
import os

app = Flask(__name__)
CORS(app)



class SNAModelGenerator:
    def __init__(self):
        self.platforms = []
        self.users = []
        self.connections = defaultdict(list)
        self.user_stats = defaultdict(lambda: {'connections': 0, 'spam_count': 0, 'total_comments': 0})
        self.channel_stats = defaultdict(lambda: {'total_comments': 0, 'spam_comments': 0})
        
    def extract_platforms_from_text(self, text):
        """Ekstrak nama platform dari teks komentar"""
        if pd.isna(text):
            return []
        
        # Pattern untuk mendeteksi platform gambling/spam
        platform_patterns = [
            r'[A-Z0-9]{3,10}88',  # Pattern seperti SGI88, MANUT88
            r'[A-Z]+[0-9]{2,3}',  # Pattern seperti ALEXIS17
            r'[A-Z]{2,}[-_]?[0-9]{2,}',  # Pattern umum platform
        ]
        
        platforms = []
        text_upper = text.upper()
        
        for pattern in platform_patterns:
            matches = re.findall(pattern, text_upper)
            platforms.extend(matches)
        
        return list(set(platforms))
    
    def classify_user_type(self, author, spam_ratio):
        """Klasifikasi user berdasarkan pola username dan aktivitas spam"""
        # Pattern untuk bot/spam account
        bot_patterns = [
            r'^[A-Z][a-z]+[A-Z][a-z]+-[a-z0-9]{3,}$',  # TerranceNoelle-o9i
            r'^[A-Z][a-z]+[A-Z][a-z]+$',  # LorrianeDotson
            r'^[A-Z][a-z]+[A-Z][a-z]+-[a-z]{2}[0-9][a-z]{2}$'  # ReshmaBanu-ml2cn
        ]
        
        for pattern in bot_patterns:
            if re.match(pattern, author):
                return 'user'
        
        return 'user'
    
    def calculate_risk_level(self, connections, spam_ratio):
        """Hitung tingkat risiko berdasarkan jumlah koneksi dan rasio spam"""
        risk_score = connections * 0.3 + spam_ratio * 100
        
        if risk_score >= 15:
            return 'high'
        elif risk_score >= 8:
            return 'medium'
        else:
            return 'low'
    
    def calculate_channel_risk_score(self, total_comments, spam_comments):
        """Hitung risk score untuk channel"""
        if total_comments == 0:
            return 0
        spam_percentage = (spam_comments / total_comments) * 100
        
        # Risk score berdasarkan persentase spam dan volume
        volume_factor = min(total_comments / 1000, 2)  # Max factor 2
        risk_score = spam_percentage * 0.1 + volume_factor
        
        return round(risk_score, 1)
    
    def process_data(self, df):
        """Proses data dan buat model SNA"""
        platform_stats = defaultdict(lambda: {'connections': set(), 'mentions': 0})
        
        for idx, row in df.iterrows():
            author = row['author']
            comment = row['komentar_clean'] if pd.notna(row['komentar_clean']) else ''
            label = row['predicted_label'] if pd.notna(row['predicted_label']) else 0
            channel = row['channel_name'] if pd.notna(row['channel_name']) else 'Unknown'
            
            # Update statistik user
            self.user_stats[author]['total_comments'] += 1
            if label == 1:
                self.user_stats[author]['spam_count'] += 1
            
            # Update statistik channel
            self.channel_stats[channel]['total_comments'] += 1
            if label == 1:
                self.channel_stats[channel]['spam_comments'] += 1
            
            # Ekstrak platform dari komentar
            platforms = self.extract_platforms_from_text(comment)
            
            for platform in platforms:
                # Tambah koneksi antara user dan platform
                self.connections[author].append(platform)
                platform_stats[platform]['connections'].add(author)
                platform_stats[platform]['mentions'] += 1
                
                # Update statistik user connections
                self.user_stats[author]['connections'] += 1
        
        # Buat node list
        network_nodes = []
        
        # Tambah platform nodes
        for platform, stats in platform_stats.items():
            connections = len(stats['connections'])
            risk_level = 'high' if connections >= 15 else 'medium' if connections >= 8 else 'low'
            
            network_nodes.append({
                'id': platform,
                'type': 'platform', 
                'connections': connections,
                'riskLevel': risk_level
            })
        
        # Tambah user nodes (hanya yang memiliki aktivitas spam)
        for user, stats in self.user_stats.items():
            if stats['total_comments'] > 0:
                spam_ratio = stats['spam_count'] / stats['total_comments']
                
                # Hanya tambahkan user yang memiliki aktivitas mencurigakan
                if spam_ratio > 0 or stats['connections'] > 5:
                    user_type = self.classify_user_type(user, spam_ratio)
                    risk_level = self.calculate_risk_level(stats['connections'], spam_ratio)
                    
                    network_nodes.append({
                        'id': user,
                        'type': user_type,
                        'connections': stats['connections'],
                        'riskLevel': risk_level
                    })
        
        # Sort berdasarkan jumlah connections (descending)
        network_nodes.sort(key=lambda x: x['connections'], reverse=True)
        
        return network_nodes
    
    def generate_edges(self, df):
        """Generate edges untuk network graph"""
        edges = []
        processed_pairs = set()
        
        for idx, row in df.iterrows():
            author = row['author']
            comment = row['komentar_clean'] if pd.notna(row['komentar_clean']) else ''
            platforms = self.extract_platforms_from_text(comment)
            
            for platform in platforms:
                pair = tuple(sorted([author, platform]))
                if pair not in processed_pairs:
                    edges.append({
                        'source': author,
                        'target': platform,
                        'weight': 1
                    })
                    processed_pairs.add(pair)
        
        return edges
    
    def get_channel_analysis(self):
        """Generate channel analysis data"""
        channel_analysis = []
        
        for channel, stats in self.channel_stats.items():
            if stats['total_comments'] > 0:  # Skip channels with no comments
                risk_score = self.calculate_channel_risk_score(
                    stats['total_comments'], 
                    stats['spam_comments']
                )
                
                channel_analysis.append({
                    'channel': channel,
                    'totalComments': stats['total_comments'],
                    'spamComments': stats['spam_comments'],
                    'riskScore': risk_score
                })
        
        # Sort by risk score descending
        channel_analysis.sort(key=lambda x: x['riskScore'], reverse=True)
        
        return channel_analysis

# Global variable to store the generator instance
sna_generator = None

def load_and_process_data():
    """Load data and create SNA model"""
    global sna_generator
    
    csv_file_path = "data_komentar_dengan_prediksi.csv"
    
    try:
        # Baca data
        df = pd.read_csv(csv_file_path)
        
        # Inisialisasi generator
        sna_generator = SNAModelGenerator()
        
        # Proses data
        network_nodes = sna_generator.process_data(df)
        network_edges = sna_generator.generate_edges(df)
        channel_analysis = sna_generator.get_channel_analysis()
        
        return {
            'networkNodes': network_nodes,
            'networkEdges': network_edges,
            'channelAnalysis': channel_analysis
        }
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

def evaluate_prediction_accuracy(df):
    """
    Melatih model dari komentar_clean dan mengevaluasi akurasi prediksi terhadap label
    """

    # Pastikan kolom penting tersedia
    if 'komentar_clean' not in df.columns or 'label' not in df.columns:
        print("Kolom 'komentar_clean' atau 'label' tidak ditemukan.")
        return None

    # Drop NA
    df = df.dropna(subset=['komentar_clean', 'label'])

    # Ambil fitur dan label
    X = df['komentar_clean']
    y = df['label'].astype(int)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model sederhana
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prediksi test set
    y_pred = model.predict(X_test)

    # Evaluasi
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    return {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

@app.route('/api/sna/evaluation', methods=['GET'])
def evaluate_model():
    """Endpoint untuk evaluasi akurasi prediksi"""
    try:
        df = pd.read_csv("/Users/yuniaameliachairunisa/Documents/SNA/data_komentar_dengan_prediksi.csv")
        result = evaluate_prediction_accuracy(df)
        if result:
            return jsonify({
                'success': True,
                'accuracy': result['accuracy'],
                'classificationReport': result['classification_report'],
                'confusionMatrix': result['confusion_matrix']
            })
        else:
            return jsonify({'success': False, 'message': 'Kolom tidak lengkap'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# Flask Routes
@app.route('/api/sna/network-nodes', methods=['GET'])
def get_network_nodes():
    """Get network nodes data"""
    try:
        data = load_and_process_data()
        if data:
            return jsonify({
                'success': True,
                'data': data['networkNodes']
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to load data'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/sna/channel-analysis', methods=['GET'])
def get_channel_analysis():
    """Get channel analysis data"""
    try:
        data = load_and_process_data()
        if data:
            return jsonify({
                'success': True,
                'data': data['channelAnalysis']
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to load data'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/sna/complete', methods=['GET'])
def get_complete_analysis():
    """Get complete SNA analysis (nodes + channels)"""
    try:
        data = load_and_process_data()
        if data:
            return jsonify({
                'success': True,
                'networkNodes': data['networkNodes'],
                'channelAnalysis': data['channelAnalysis'],
                'networkEdges': data['networkEdges']
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to load data'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/sna/export-json', methods=['GET'])
def export_to_json():
    """Export data as JSON files for React"""
    try:
        data = load_and_process_data()
        if data:
            # Create exports directory if not exists
            os.makedirs('exports', exist_ok=True)
            
            # Save network nodes
            with open('exports/networkNodes.json', 'w', encoding='utf-8') as f:
                json.dump(data['networkNodes'], f, indent=2, ensure_ascii=False)
            
            # Save channel analysis
            with open('exports/channelAnalysis.json', 'w', encoding='utf-8') as f:
                json.dump(data['channelAnalysis'], f, indent=2, ensure_ascii=False)
            
            # Save network edges  
            with open('exports/networkEdges.json', 'w', encoding='utf-8') as f:
                json.dump(data['networkEdges'], f, indent=2, ensure_ascii=False)
            
            return jsonify({
                'success': True,
                'message': 'JSON files exported successfully',
                'files': [
                    'exports/networkNodes.json',
                    'exports/channelAnalysis.json', 
                    'exports/networkEdges.json'
                ]
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to load data'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/')
def home():
    return "Hello from Railway!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # ambil dari Railway
    app.run(host='0.0.0.0', port=port)
