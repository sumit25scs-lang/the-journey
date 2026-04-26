"""
Retenza – Student Dropout Risk Prediction System
Flask Backend with Real Random Forest ML Model
"""

import os
import io
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template, send_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             precision_score, recall_score, f1_score)

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ================================================================
# CONFIGURATION
# ================================================================
BASE_DIR    = os.path.dirname(__file__)
MODEL_DIR   = os.path.join(BASE_DIR, 'model')
MODEL_PATH  = os.path.join(MODEL_DIR, 'dropout_model.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

FEATURES = [
    'completion_rate_1', 'completion_rate_2',
    'approved_1sem',     'approved_2sem',
    'grade_1sem',        'grade_2sem',
    'tuition_fees',      'financial_risk',
    'scholarship',       'debt',
    'attendance',        'study_hrs',
    'exam_prep',         'employed',
    'childcare',         'displaced',
    'age',
]

FEATURE_DISPLAY_NAMES = {
    'completion_rate_2': 'Completion Rate (Sem 2)',
    'completion_rate_1': 'Completion Rate (Sem 1)',
    'approved_2sem':     'Approved Courses (Sem 2)',
    'grade_2sem':        'Average Grades (Sem 2)',
    'grade_1sem':        'Average Grades (Sem 1)',
    'financial_risk':    'Financial Risk',
    'tuition_fees':      'Tuition Fees Status',
    'attendance':        'Class Attendance',
    'approved_1sem':     'Approved Courses (Sem 1)',
    'study_hrs':         'Weekly Study Hours',
    'age':               'Age of Enrollment',
    'scholarship':       'Scholarship Holder',
    'exam_prep':         'Exam Prep Level',
    'employed':          'Employment Status',
    'debt':              'Outstanding Debt',
    'childcare':         'Childcare Responsibilities',
    'displaced':         'Displaced / Refugee Status',
}

# Global model + metrics
_model   = None
_metrics = None


# ================================================================
# SYNTHETIC DATA GENERATION
# ================================================================
def generate_training_data(n_total: int = 3630, dropout_rate: float = 0.3915,
                            seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_out = int(n_total * dropout_rate)   # ~1421 dropouts
    n_in  = n_total - n_out               # ~2209 enrolled

    def make_class(n, dropout: bool) -> pd.DataFrame:
        if dropout:
            cr1  = np.clip(rng.beta(2,   5,   n), 0.0, 1.0)
            cr2  = np.clip(rng.beta(1.5, 5,   n), 0.0, 1.0)
            ap1  = np.clip(rng.poisson(2.0, n),   0, 8)
            ap2  = np.clip(rng.poisson(1.5, n),   0, 8)
            g1   = np.clip(rng.normal(9.0, 3.0, n),  0, 20)
            g2   = np.clip(rng.normal(8.0, 3.0, n),  0, 20)
            tuit = rng.binomial(1, 0.25, n)
            finr = rng.binomial(1, 0.55, n)
            sch  = rng.binomial(1, 0.10, n)
            debt = rng.binomial(1, 0.50, n)
            att  = rng.choice([1,2,3,4], n, p=[0.35,0.35,0.20,0.10])
            stdy = rng.choice([1,2,3,4], n, p=[0.40,0.35,0.18,0.07])
            exam = rng.choice([1,2,3],   n, p=[0.50,0.35,0.15])
            emp  = rng.choice([0,1,2],   n, p=[0.25,0.40,0.35])
            chld = rng.binomial(1, 0.30, n)
            disp = rng.binomial(1, 0.15, n)
            age  = np.clip(rng.normal(26, 7, n), 17, 60)
        else:
            cr1  = np.clip(rng.beta(8, 2, n),   0.0, 1.0)
            cr2  = np.clip(rng.beta(8, 2, n),   0.0, 1.0)
            ap1  = np.clip(rng.poisson(5, n)+2,  0, 8)
            ap2  = np.clip(rng.poisson(5, n)+2,  0, 8)
            g1   = np.clip(rng.normal(14.0, 2.5, n), 0, 20)
            g2   = np.clip(rng.normal(14.0, 2.5, n), 0, 20)
            tuit = rng.binomial(1, 0.88, n)
            finr = rng.binomial(1, 0.10, n)
            sch  = rng.binomial(1, 0.35, n)
            debt = rng.binomial(1, 0.10, n)
            att  = rng.choice([1,2,3,4], n, p=[0.05,0.15,0.40,0.40])
            stdy = rng.choice([1,2,3,4], n, p=[0.05,0.20,0.45,0.30])
            exam = rng.choice([1,2,3],   n, p=[0.10,0.40,0.50])
            emp  = rng.choice([0,1,2],   n, p=[0.55,0.35,0.10])
            chld = rng.binomial(1, 0.10, n)
            disp = rng.binomial(1, 0.05, n)
            age  = np.clip(rng.normal(21, 3, n), 17, 40)

        return pd.DataFrame({
            'completion_rate_1': cr1, 'completion_rate_2': cr2,
            'approved_1sem': ap1.astype(int), 'approved_2sem': ap2.astype(int),
            'grade_1sem': g1, 'grade_2sem': g2,
            'tuition_fees': tuit, 'financial_risk': finr,
            'scholarship': sch, 'debt': debt,
            'attendance': att, 'study_hrs': stdy,
            'exam_prep': exam, 'employed': emp,
            'childcare': chld, 'displaced': disp,
            'age': age,
            'dropout': int(dropout),
        })

    df = pd.concat([make_class(n_out, True), make_class(n_in, False)],
                   ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ================================================================
# MODEL TRAINING
# ================================================================
def train_model():
    global _model, _metrics
    os.makedirs(MODEL_DIR, exist_ok=True)

    print('[Retenza] Generating 3,630-record synthetic training dataset...')
    df = generate_training_data()
    X, y = df[FEATURES], df['dropout']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f'[Retenza] Training Random Forest on {len(X_train):,} samples...')
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f'[Retenza] Accuracy: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}')

    fi_list = sorted(
        [{'feature': FEATURE_DISPLAY_NAMES.get(f, f),
          'importance': round(float(imp)*100, 2)}
         for f, imp in zip(FEATURES, rf.feature_importances_)],
        key=lambda x: -x['importance']
    )

    metrics_data = {
        'accuracy':      round(acc*100, 2),
        'auc':           round(auc, 4),
        'precision':     round(prec*100, 2),
        'recall':        round(rec*100, 2),
        'f1':            round(f1*100, 2),
        'training_size': len(X_train),
        'test_size':     len(X_test),
        'n_features':    len(FEATURES),
        'dropout_rate':  round(float(y.mean())*100, 2),
        'n_estimators':  200,
        'feature_importances': fi_list,
    }

    joblib.dump(rf, MODEL_PATH)
    with open(METRICS_PATH, 'w') as fh:
        json.dump(metrics_data, fh, indent=2)

    _model, _metrics = rf, metrics_data
    return rf, metrics_data


def load_or_train():
    global _model, _metrics
    if os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
        print('[Retenza] Loading saved model…')
        _model = joblib.load(MODEL_PATH)
        with open(METRICS_PATH) as fh:
            _metrics = json.load(fh)
        print(f"[Retenza] Model ready – Accuracy: {_metrics['accuracy']}%")
    else:
        train_model()


# ================================================================
# RISK FACTOR ANALYSIS (Python mirror of the JS logic)
# ================================================================
def analyze_factors(data: dict):
    factors, breakdown = [], {}

    cr1  = float(data.get('cr1', 0.8))
    cr2  = float(data.get('cr2', 0.8))
    g1   = float(data.get('g1', 14))
    g2   = float(data.get('g2', 14))
    ap2  = int(float(data.get('ap2', 5)))

    # Academic (max 55)
    ac = 0
    if cr2 < 0.5:   ac += 18; factors.append({'name': 'Very low completion rate (Sem 2)', 'sev': 'High',   'icon': '📉'})
    elif cr2 < 0.7: ac += 10; factors.append({'name': 'Below average completion rate (Sem 2)', 'sev': 'Medium', 'icon': '📊'})
    if cr1 < 0.5:   ac += 14; factors.append({'name': 'Very low completion rate (Sem 1)', 'sev': 'High',   'icon': '📉'})
    elif cr1 < 0.7: ac += 8;  factors.append({'name': 'Below average completion rate (Sem 1)', 'sev': 'Medium', 'icon': '📊'})
    if g2 < 8:      ac += 10; factors.append({'name': 'Very low average grade (Sem 2)', 'sev': 'High',     'icon': '📝'})
    elif g2 < 11:   ac += 5;  factors.append({'name': 'Below passing grade (Sem 2)', 'sev': 'Medium',      'icon': '📝'})
    if g1 < 8:      ac += 7;  factors.append({'name': 'Very low average grade (Sem 1)', 'sev': 'High',     'icon': '📝'})
    elif g1 < 11:   ac += 3;  factors.append({'name': 'Below passing grade (Sem 1)', 'sev': 'Medium',      'icon': '📝'})
    if g2 < g1-3:   ac += 4;  factors.append({'name': 'Declining grade trend Sem1→Sem2', 'sev': 'Medium',  'icon': '📉'})
    if ap2 < 3:     ac += 8;  factors.append({'name': 'Few approved courses (Sem 2)', 'sev': 'High',       'icon': '❌'})
    elif ap2 < 4:   ac += 4;  factors.append({'name': 'Below average approved courses (Sem 2)', 'sev': 'Medium', 'icon': '⚠️'})
    breakdown['Academic'] = {'score': min(ac, 55), 'max': 55}

    # Financial (max 20)
    fin = 0
    if str(data.get('finrisk', '0')) == '1': fin += 8; factors.append({'name': 'Financial risk detected', 'sev': 'High',    'icon': '💸'})
    if str(data.get('tuition', '1')) == '0': fin += 8; factors.append({'name': 'Tuition fees not up to date', 'sev': 'High', 'icon': '💰'})
    if str(data.get('debt',    '0')) == '1': fin += 4; factors.append({'name': 'Outstanding debt present', 'sev': 'Medium',  'icon': '💳'})
    breakdown['Financial'] = {'score': min(fin, 20), 'max': 20}

    # Engagement (max 15)
    eng = 0
    att  = int(float(data.get('attendance', 3)))
    stdy = int(float(data.get('study', 3)))
    exam = int(float(data.get('exam', 2)))
    if att == 1:  eng += 6; factors.append({'name': 'Poor class attendance (<60%)', 'sev': 'High',         'icon': '🚫'})
    elif att == 2:eng += 3; factors.append({'name': 'Fair class attendance (60-74%)', 'sev': 'Medium',     'icon': '⚠️'})
    if stdy == 1: eng += 5; factors.append({'name': 'Very few study hours (<5/week)', 'sev': 'High',       'icon': '⏱️'})
    elif stdy==2: eng += 2; factors.append({'name': 'Limited study hours (5-10/week)', 'sev': 'Low',       'icon': '⏱️'})
    if exam == 1: eng += 4; factors.append({'name': 'Low exam preparation level', 'sev': 'Medium',         'icon': '📚'})
    breakdown['Engagement'] = {'score': min(eng, 15), 'max': 15}

    # Personal (max 10)
    per = 0
    age  = int(float(data.get('age', 20)))
    emp  = int(float(data.get('employed', 0)))
    if age > 30:                               per += 3; factors.append({'name': 'Non-traditional age (>30)', 'sev': 'Low',          'icon': '👤'})
    if emp == 2:                               per += 4; factors.append({'name': 'Full-time work while studying', 'sev': 'High',      'icon': '💼'})
    elif emp == 1:                             per += 2; factors.append({'name': 'Part-time work while studying', 'sev': 'Low',       'icon': '💼'})
    if str(data.get('childcare', '0')) == '1': per += 2; factors.append({'name': 'Childcare responsibilities', 'sev': 'Low',         'icon': '👶'})
    if str(data.get('displaced', '0')) == '1': per += 1; factors.append({'name': 'Displaced / refugee status', 'sev': 'Low',         'icon': '🌍'})
    breakdown['Personal'] = {'score': min(per, 10), 'max': 10}

    return factors, breakdown


def build_interventions(breakdown: dict) -> list:
    has_academic   = breakdown['Academic']['score'] > 10
    has_financial  = breakdown['Financial']['score'] > 0
    has_engagement = breakdown['Engagement']['score'] > 0
    has_personal   = breakdown['Personal']['score'] > 3

    interventions = []
    if has_academic or has_engagement:
        interventions.append({
            'dept': 'Academic Support Services',
            'email': 'academic.support@university.edu',
            'phone': '+1 (555) 123-4581',
            'color': 'rgba(99,102,241,0.15)',
            'iconColor': 'var(--primary-light)',
            'icon': 'fa fa-book',
            'actions': ['Schedule academic advisor meeting',
                        'Enroll in tutoring programs',
                        'Join structured study groups',
                        'Review course load balance'],
        })
    if has_financial:
        interventions.append({
            'dept': 'Financial Aid Office',
            'email': 'financialaid@university.edu',
            'phone': '+1 (555) 123-4582',
            'color': 'rgba(16,185,129,0.15)',
            'iconColor': 'var(--success)',
            'icon': 'fa fa-dollar-sign',
            'actions': ['Apply for emergency financial assistance',
                        'Review payment plan options',
                        'Explore scholarship opportunities',
                        'Debt counseling session'],
        })
    if has_engagement:
        interventions.append({
            'dept': 'Student Counseling Services',
            'email': 'counseling@university.edu',
            'phone': '+1 (555) 123-4583',
            'color': 'rgba(245,158,11,0.15)',
            'iconColor': 'var(--warning)',
            'icon': 'fa fa-comments',
            'actions': ['Time management workshops',
                        'Study skills coaching session',
                        'Peer mentoring program',
                        'Goal-setting consultation'],
        })
    if has_personal:
        interventions.append({
            'dept': 'Wellness Center',
            'email': 'wellness@university.edu',
            'phone': '+1 (555) 123-4584',
            'color': 'rgba(244,114,182,0.15)',
            'iconColor': 'var(--accent2)',
            'icon': 'fa fa-heart-pulse',
            'actions': ['Wellness consultation',
                        'Work-life balance assessment',
                        'Support group referral',
                        'Resource navigation assistance'],
        })
    if not interventions:
        interventions.append({
            'dept': 'Student Success Center',
            'email': 'success@university.edu',
            'phone': '+1 (555) 123-4580',
            'color': 'rgba(16,185,129,0.15)',
            'iconColor': 'var(--success)',
            'icon': 'fa fa-star',
            'actions': ['Continue current academic path',
                        'Explore advanced courses / research',
                        'Mentoring program participation',
                        'Career development services'],
        })
    return interventions


def row_to_features(data: dict) -> pd.DataFrame:
    """Build a one-row DataFrame from form / CSV data."""
    return pd.DataFrame([{
        'completion_rate_1': float(data.get('cr1', 0.8)),
        'completion_rate_2': float(data.get('cr2', 0.8)),
        'approved_1sem':     int(float(data.get('ap1', 5))),
        'approved_2sem':     int(float(data.get('ap2', 5))),
        'grade_1sem':        float(data.get('g1', 14)),
        'grade_2sem':        float(data.get('g2', 14)),
        'tuition_fees':      int(float(data.get('tuition', 1))),
        'financial_risk':    int(float(data.get('finrisk', 0))),
        'scholarship':       int(float(data.get('scholarship', 0))),
        'debt':              int(float(data.get('debt', 0))),
        'attendance':        int(float(data.get('attendance', 3))),
        'study_hrs':         int(float(data.get('study', 3))),
        'exam_prep':         int(float(data.get('exam', 2))),
        'employed':          int(float(data.get('employed', 0))),
        'childcare':         int(float(data.get('childcare', 0))),
        'displaced':         int(float(data.get('displaced', 0))),
        'age':               float(data.get('age', 20)),
    }])


# ================================================================
# ROUTES
# ================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data   = request.get_json(force=True)
        feat   = row_to_features(data)
        probs  = _model.predict_proba(feat[FEATURES])[0]
        prob   = float(probs[1])
        score  = int(round(prob * 100))
        risk   = 'High' if score >= 60 else 'Medium' if score >= 40 else 'Low'
        conf   = round(float(max(probs)), 3)

        factors, breakdown = analyze_factors(data)
        interventions = build_interventions(breakdown)

        return jsonify({
            'success': True,
            'score': score,
            'riskLevel': risk,
            'probability': round(prob, 4),
            'confidence': conf,
            'factors': factors,
            'breakdown': breakdown,
            'interventions': interventions,
            'model': 'Random Forest (200 trees)',
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/bulk-predict', methods=['POST'])
def bulk_predict():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        fname = file.filename.lower()
        if fname.endswith('.csv'):
            df = pd.read_csv(file)
        elif fname.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'success': False, 'error': 'Unsupported format. Use .csv or .xlsx'}), 400

        results = []
        for _, row in df.iterrows():
            data = {
                'cr1':        row.get('completion_rate_1', 0.8),
                'cr2':        row.get('completion_rate_2', 0.8),
                'ap1':        row.get('approved_1sem', 5),
                'ap2':        row.get('approved_2sem', 5),
                'g1':         row.get('grade_1sem', 14),
                'g2':         row.get('grade_2sem', 14),
                'tuition':    row.get('tuition_fees', 1),
                'finrisk':    row.get('financial_risk', 0),
                'scholarship':row.get('scholarship', 0),
                'debt':       row.get('debt', 0),
                'attendance': row.get('attendance', 3),
                'study':      row.get('study_hrs', 3),
                'exam':       row.get('exam_prep', 2),
                'employed':   row.get('employed', 0),
                'childcare':  row.get('childcare', 0),
                'displaced':  row.get('displaced', 0),
                'age':        row.get('age', 20),
            }
            feat  = row_to_features(data)
            probs = _model.predict_proba(feat[FEATURES])[0]
            prob  = float(probs[1])
            score = int(round(prob * 100))
            risk  = 'High' if score >= 60 else 'Medium' if score >= 40 else 'Low'

            factors, breakdown = analyze_factors(data)
            results.append({
                'name':        str(row.get('student_name', row.get('name', 'Unknown'))),
                'sid':         str(row.get('student_id',   row.get('id', ''))),
                'score':       score,
                'riskLevel':   risk,
                'probability': round(prob, 4),
                'age':         data['age'],
                'factors':     factors,
                'breakdown':   breakdown,
            })

        high = sum(1 for r in results if r['riskLevel'] == 'High')
        med  = sum(1 for r in results if r['riskLevel'] == 'Medium')
        low  = sum(1 for r in results if r['riskLevel'] == 'Low')

        return jsonify({
            'success': True,
            'results': results,
            'summary': {'total': len(results), 'high': high, 'medium': med, 'low': low},
        })
    except Exception as exc:
        import traceback
        return jsonify({'success': False,
                        'error': str(exc),
                        'trace': traceback.format_exc()}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    try:
        fi = sorted(
            [{'feature': FEATURE_DISPLAY_NAMES.get(f, f),
              'importance': round(float(imp)*100, 2)}
             for f, imp in zip(FEATURES, _model.feature_importances_)],
            key=lambda x: -x['importance']
        )
        return jsonify({'success': True, 'metrics': _metrics, 'feature_importances': fi})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/export/excel', methods=['POST'])
def export_excel():
    try:
        results = request.get_json(force=True).get('results', [])
        records = [{
            'Student Name':   r.get('name', ''),
            'Student ID':     r.get('sid', ''),
            'Risk Score %':   r.get('score', 0),
            'Risk Level':     r.get('riskLevel', ''),
            'Probability':    r.get('probability', 0),
            'Age':            r.get('age', ''),
            'Key Risk Factors': '; '.join(f['name'] for f in r.get('factors', [])[:3]),
        } for r in results]

        buf = io.BytesIO()
        pd.DataFrame(records).to_excel(buf, index=False,
                                        sheet_name='Risk Assessment',
                                        engine='openpyxl')
        buf.seek(0)
        return send_file(buf,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True,
                         download_name='Retenza_Bulk_Results.xlsx')
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Force model retraining (admin use)."""
    try:
        rf, m = train_model()
        return jsonify({'success': True, 'accuracy': m['accuracy'], 'auc': m['auc']})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print('=' * 54)
    print('  Retenza - Student Dropout Risk Prediction System')
    print('=' * 54)
    load_or_train()
    print('\n  Server -> http://localhost:5000\n  Press Ctrl+C to stop.\n')
    app.run(debug=False, host='0.0.0.0', port=5000)
