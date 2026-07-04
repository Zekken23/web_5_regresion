from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


app = Flask(__name__)

DATA_PATH = 'AgroFertilizerLoss.csv'
MODEL_PATH = 'xgboost_regression_model.pkl'
BATCH_RESULT_PATH = os.path.join('static', 'batch_result.csv')

model_data = joblib.load(MODEL_PATH)
model = model_data['model']
features = model_data['features']
encoders = model_data['encoders']


def prepare_batch_features(df_batch):
    missing_features = [col for col in features if col not in df_batch.columns]
    if missing_features:
        raise ValueError("Kolom fitur tidak ditemukan: " + ", ".join(missing_features))

    df_model = df_batch[features].copy()
    cat_features = set(encoders.keys())

    for col in features:
        if col in cat_features:
            le = encoders[col]
            df_model[col] = df_model[col].fillna(le.classes_[0]).astype(str)
            df_model[col] = df_model[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df_model[col] = le.transform(df_model[col])
        else:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0.0)

    return df_model


def build_model_frame(df_raw):
    y = df_raw['Fertilizer_Loss_Percentage']
    columns_to_drop = [
        'Total_Fertilizer_Loss_kg_ha',
        'Nitrogen_Loss',
        'Phosphorus_Loss',
        'Potassium_Loss',
        'Fertilizer_Loss_Percentage'
    ]
    X = df_raw.drop(columns=columns_to_drop)
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    X_encoded = X_clean.copy()
    for col, le in encoders.items():
        X_encoded[col] = X_encoded[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        X_encoded[col] = le.transform(X_encoded[col])

    return X_clean, X_encoded, y_clean, columns_to_drop


def rounded(value, digits=4):
    return round(float(value), digits) if pd.notna(value) else 0


def get_dashboard_data():
    df_raw = pd.read_csv(DATA_PATH)
    X_clean, X_encoded, y_clean, columns_to_drop = build_model_frame(df_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_clean, test_size=0.2, random_state=42
    )
    y_pred = model.predict(X_test)
    residuals = y_test.to_numpy() - y_pred

    metrics = {
        'mae': rounded(mean_absolute_error(y_test, y_pred)),
        'mse': rounded(mean_squared_error(y_test, y_pred)),
        'rmse': rounded(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': rounded(r2_score(y_test, y_pred)),
    }

    numeric_features = [col for col in features if col not in encoders]
    target_stats = {
        'mean': rounded(y_clean.mean(), 2),
        'median': rounded(y_clean.median(), 2),
        'min': rounded(y_clean.min(), 2),
        'max': rounded(y_clean.max(), 2),
        'std': rounded(y_clean.std(), 2),
    }

    bins = pd.cut(y_clean, bins=6)
    target_distribution = [
        {'label': str(interval), 'count': int(count)}
        for interval, count in bins.value_counts().sort_index().items()
    ]

    correlation_rows = []
    for col in numeric_features:
        corr = pd.to_numeric(X_clean[col], errors='coerce').corr(y_clean)
        if pd.notna(corr):
            correlation_rows.append({
                'feature': col,
                'correlation': rounded(corr, 4),
                'width': rounded(min(abs(corr) * 100, 100), 1),
                'strength': 'Kuat' if abs(corr) >= 0.5 else 'Sedang' if abs(corr) >= 0.25 else 'Rendah'
            })
    correlation_rows = sorted(correlation_rows, key=lambda item: abs(item['correlation']), reverse=True)[:10]

    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        pairs = sorted(zip(features, model.feature_importances_), key=lambda item: item[1], reverse=True)
        feature_importance = [
            {'feature': feature, 'importance': rounded(score, 5)}
            for feature, score in pairs[:12]
        ]

    categorical_rows = []
    for col in encoders:
        counts = X_clean[col].value_counts().head(5)
        categorical_rows.append({
            'feature': col,
            'unique': int(X_clean[col].nunique()),
            'top_values': [{'name': str(name), 'count': int(count)} for name, count in counts.items()]
        })

    residual_sample = [
        {
            'actual': rounded(actual, 2),
            'predicted': rounded(predicted, 2),
            'error': rounded(error, 2)
        }
        for actual, predicted, error in zip(y_test.head(40), y_pred[:40], residuals[:40])
    ]

    return {
        'summary': {
            'total_rows': int(df_raw.shape[0]),
            'total_columns': int(df_raw.shape[1]),
            'clean_rows': int(X_clean.shape[0]),
            'feature_count': len(features),
            'numeric_count': len(numeric_features),
            'categorical_count': len(encoders),
            'train_rows': int(X_train.shape[0]),
            'test_rows': int(X_test.shape[0]),
            'model_name': 'XGBoost Regressor',
            'target': 'Fertilizer_Loss_Percentage',
        },
        'target_stats': target_stats,
        'metrics': metrics,
        'columns_to_drop': columns_to_drop,
        'raw_head': df_raw.head(8).to_dict(orient='records'),
        'clean_head': pd.concat([X_clean, y_clean], axis=1).head(8).to_dict(orient='records'),
        'target_distribution': target_distribution,
        'correlation_rows': correlation_rows,
        'feature_importance': feature_importance,
        'categorical_rows': categorical_rows,
        'residual_sample': residual_sample,
        'pipeline': [
            {'step': 'Load Dataset', 'desc': 'Baca AgroFertilizerLoss.csv sebagai sumber training.', 'state': 'done'},
            {'step': 'Target Regression', 'desc': 'Target memakai Fertilizer_Loss_Percentage.', 'state': 'done'},
            {'step': 'Leakage Removal', 'desc': 'Drop loss total dan komponen N/P/K sebelum training.', 'state': 'done'},
            {'step': 'Cleaning', 'desc': 'Drop missing rows sesuai train_regression.py.', 'state': 'done'},
            {'step': 'Encoding', 'desc': 'LabelEncoder untuk fitur kategorik tersimpan di artifact.', 'state': 'done'},
            {'step': 'Prediction Runtime', 'desc': 'Form dan batch memakai artifact XGBoost tersimpan.', 'state': 'now'},
        ],
        'model_cards': [
            {'name': 'Regressor', 'meta': 'XGBoost', 'desc': 'Model utama untuk estimasi persentase kehilangan pupuk.'},
            {'name': 'Split', 'meta': '80:20', 'desc': 'train_test_split random_state=42 mengikuti file training.'},
            {'name': 'Target', 'meta': '% Loss', 'desc': 'Output berupa Fertilizer_Loss_Percentage.'},
            {'name': 'Runtime', 'meta': 'Flask', 'desc': 'Single prediction dan batch CSV dari model pickle.'},
        ],
    }


def render_dashboard(active_page='overview', prediction=None, batch_result=None, batch_error=None):
    cat_features = list(encoders.keys())
    num_features = [f for f in features if f not in cat_features]
    return render_template(
        'index.html',
        dashboard=get_dashboard_data(),
        features=features,
        cat_features=cat_features,
        num_features=num_features,
        encoders=encoders,
        active_page=active_page,
        prediction=prediction,
        batch_result=batch_result,
        batch_error=batch_error,
    )


@app.route('/')
def dashboard():
    return render_dashboard()


@app.route('/single_test', methods=['GET', 'POST'])
def single_test():
    prediction = None
    if request.method == 'POST':
        cat_features = set(encoders.keys())
        input_data = {}
        for feature in features:
            value = request.form.get(feature)
            input_data[feature] = value if feature in cat_features else (float(value) if value else 0.0)

        df_input = pd.DataFrame([input_data])
        for col, le in encoders.items():
            df_input[col] = df_input[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df_input[col] = le.transform(df_input[col])

        prediction = rounded(model.predict(df_input[features])[0], 4)

    return render_dashboard(active_page='prediction', prediction=prediction)


@app.route('/batch_test', methods=['GET', 'POST'])
def batch_test():
    batch_result = None
    batch_error = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            batch_error = "Pilih file CSV terlebih dahulu."
        elif not file.filename.lower().endswith('.csv'):
            batch_error = "File harus berformat CSV."
        else:
            try:
                df_batch = pd.read_csv(file)
                df_display = df_batch.copy()
                df_model = prepare_batch_features(df_batch)
                df_display['Prediksi_Fertilizer_Loss_%'] = np.round(model.predict(df_model), 4)

                os.makedirs(os.path.dirname(BATCH_RESULT_PATH), exist_ok=True)
                df_display.to_csv(BATCH_RESULT_PATH, index=False)
                chart_rows = []
                for _, row in df_display.head(40).iterrows():
                    chart_rows.append({
                        'predicted': rounded(row['Prediksi_Fertilizer_Loss_%'], 4)
                    })

                batch_result = {
                    'rows': int(df_display.shape[0]),
                    'columns': int(df_display.shape[1]),
                    'preview': df_display.head(10).to_dict(orient='records'),
                    'chart_rows': chart_rows,
                }
            except Exception as exc:
                batch_error = f"Gagal memproses file CSV: {exc}"

    return render_dashboard(active_page='batch', batch_result=batch_result, batch_error=batch_error)


@app.route('/download')
def download():
    return send_file(BATCH_RESULT_PATH, as_attachment=True)


if __name__ == '__main__':
    app.run(port=5010, debug=True)
