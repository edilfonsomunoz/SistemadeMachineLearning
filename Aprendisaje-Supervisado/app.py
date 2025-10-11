from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt
import json
from datetime import datetime, timedelta
import secrets
import pandas as pd
import numpy as np
from io import BytesIO
import logging

logging.basicConfig(level=logging.ERROR)

from ml_models.regression import RegressionModels
from ml_models.logistic import LogisticModels
from ml_models.decision_tree import DecisionTreeModels
from ml_models.report_generator import ReportGenerator
from ml_models.data_handler import DataHandler

load_dotenv()

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

DATABASE_URL = os.getenv('DATABASE_URL')

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            student_code VARCHAR(6) NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            name VARCHAR(255) NOT NULL,
            data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            token VARCHAR(255) UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.json
        email = data.get('email')
        name = data.get('name')
        student_code = data.get('student_code')
        password = data.get('password')
        
        if not email or not name or not student_code or not password:
            return jsonify({'error': 'Todos los campos son requeridos'}), 400
        
        if len(student_code) != 6 or not student_code.isdigit():
            return jsonify({'error': 'El código de estudiante debe tener 6 dígitos'}), 400
        
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT INTO users (email, name, student_code, password_hash) VALUES (%s, %s, %s, %s) RETURNING id',
                (email, name, student_code, password_hash)
            )
            user_id = cursor.fetchone()[0]
            conn.commit()
            
            return jsonify({
                'message': 'Usuario registrado exitosamente',
                'user_id': user_id
            }), 201
        except psycopg2.IntegrityError:
            return jsonify({'error': 'El email ya está registrado'}), 400
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logging.error(f"Error in register: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error en el registro'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email y contraseña son requeridos'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return jsonify({'error': 'Credenciales inválidas'}), 401
        
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=7)
        
        cursor.execute(
            'INSERT INTO sessions (user_id, token, expires_at) VALUES (%s, %s, %s)',
            (user['id'], token, expires_at)
        )
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': 'Login exitoso',
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'name': user['name'],
                'student_code': user['student_code']
            }
        }), 200
    except Exception as e:
        logging.error(f"Error in login: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error en el inicio de sesión'}), 500

def verify_token(token):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute('''
        SELECT sessions.*, users.id as user_id, users.email, users.name, users.student_code
        FROM sessions
        JOIN users ON sessions.user_id = users.id
        WHERE sessions.token = %s AND sessions.expires_at > NOW()
    ''', (token,))
    
    session = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return session

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_token(token)
        
        if not session:
            return jsonify({'error': 'No autorizado'}), 401
        
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo'}), 400
        
        file = request.files['file']
        filename = file.filename
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Formato de archivo no soportado'}), 400
        
        data_dict = df.to_dict(orient='records')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO datasets (user_id, name, data) VALUES (%s, %s, %s) RETURNING id',
            (session['user_id'], filename, json.dumps(data_dict))
        )
        dataset_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': 'Datos cargados exitosamente',
            'dataset_id': dataset_id,
            'columns': list(df.columns),
            'rows': len(df),
            'preview': df.head(10).to_dict(orient='records')
        }), 200
    except Exception as e:
        logging.error(f"Error in upload_data: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error al procesar el archivo'}), 500

@app.route('/api/generate-random-data', methods=['POST'])
def generate_random_data():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_token(token)
        
        if not session:
            return jsonify({'error': 'No autorizado'}), 401
        
        data = request.json
        rows = data.get('rows', 100)
        columns = data.get('columns', 5)
        
        df = DataHandler.generate_random_data(rows, columns)
        data_dict = df.to_dict(orient='records')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO datasets (user_id, name, data) VALUES (%s, %s, %s) RETURNING id',
            (session['user_id'], f'random_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}', json.dumps(data_dict))
        )
        dataset_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': 'Datos generados exitosamente',
            'dataset_id': dataset_id,
            'columns': list(df.columns),
            'rows': len(df),
            'preview': df.head(10).to_dict(orient='records')
        }), 200
    except Exception as e:
        logging.error(f"Error in generate_random_data: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error al generar datos'}), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_token(token)
        
        if not session:
            return jsonify({'error': 'No autorizado'}), 401
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            'SELECT id, name, created_at FROM datasets WHERE user_id = %s ORDER BY created_at DESC',
            (session['user_id'],)
        )
        datasets = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'datasets': datasets}), 200
    except Exception as e:
        logging.error(f"Error in get_datasets: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error al obtener datasets'}), 500

@app.route('/api/regression', methods=['POST'])
def run_regression():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_token(token)
        
        if not session:
            return jsonify({'error': 'No autorizado'}), 401
        
        data = request.json
        dataset_id = data.get('dataset_id')
        model_type = data.get('model_type')
        target_column = data.get('target_column')
        feature_columns = data.get('feature_columns', [])
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('SELECT data FROM datasets WHERE id = %s AND user_id = %s', (dataset_id, session['user_id']))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            return jsonify({'error': 'Dataset no encontrado'}), 404
        
        df = pd.DataFrame(result['data'])
        
        regression_models = RegressionModels()
        result = regression_models.run_model(df, model_type, target_column, feature_columns)
        
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error in run_regression: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error al ejecutar el modelo de regresión'}), 500

@app.route('/api/logistic', methods=['POST'])
def run_logistic():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_token(token)
        
        if not session:
            return jsonify({'error': 'No autorizado'}), 401
        
        data = request.json
        dataset_id = data.get('dataset_id')
        target_column = data.get('target_column')
        feature_columns = data.get('feature_columns', [])
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('SELECT data FROM datasets WHERE id = %s AND user_id = %s', (dataset_id, session['user_id']))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            return jsonify({'error': 'Dataset no encontrado'}), 404
        
        df = pd.DataFrame(result['data'])
        
        logistic_models = LogisticModels()
        result = logistic_models.run_model(df, target_column, feature_columns)
        
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error in run_logistic: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error al ejecutar el modelo logístico'}), 500

@app.route('/api/decision-tree', methods=['POST'])
def run_decision_tree():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_token(token)
        
        if not session:
            return jsonify({'error': 'No autorizado'}), 401
        
        data = request.json
        dataset_id = data.get('dataset_id')
        model_type = data.get('model_type', 'cart')
        target_column = data.get('target_column')
        feature_columns = data.get('feature_columns', [])
        max_depth = data.get('max_depth', None)
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('SELECT data FROM datasets WHERE id = %s AND user_id = %s', (dataset_id, session['user_id']))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            return jsonify({'error': 'Dataset no encontrado'}), 404
        
        df = pd.DataFrame(result['data'])
        
        dt_models = DecisionTreeModels()
        result = dt_models.run_model(df, model_type, target_column, feature_columns, max_depth)
        
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error in run_decision_tree: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error al ejecutar el árbol de decisión'}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_token(token)
        
        if not session:
            return jsonify({'error': 'No autorizado'}), 401
        
        data = request.json
        report_data = data.get('report_data', {})
        
        report_gen = ReportGenerator()
        pdf_buffer = report_gen.generate_pdf(report_data, session)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'reporte_ml_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    except Exception as e:
        logging.error(f"Error in generate_report: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error al generar el reporte'}), 500

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
    try:
        return send_from_directory('static', path)
    except:
        return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    init_db()
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
