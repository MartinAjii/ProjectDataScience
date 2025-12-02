import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import PartialDependenceDisplay


st.title("📊 Random Forest - High Blood Pressure Prediction")
st.write("Aplikasi Streamlit yang menampilkan seluruh output pemrosesan data, training model, evaluasi, dan prediksi.")

tab1, tab2, tab3 = st.tabs(['Data Preparation dan Modeling', 'Visualisasi', 'Input Data Baru'])
with tab1:
    st.write('Data Preparation dan Modeling')

    # 1. LOAD DATA
    st.header("1. Load Dataset")

    df = pd.read_csv("smoking_health_data_final.csv")

    # Menampilkan info df
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    st.subheader("Dataset Info")
    st.text(info_str)

    st.subheader("Sample Data")
    st.write(df.sample(5))


    # 2. DATA PREPARATION
    st.header("2. Data Preparation")

    # Missing value
    if 'cigs_per_day' in df.columns:
        df['cigs_per_day'] = df['cigs_per_day'].fillna(df['cigs_per_day'].median())
    if 'chol' in df.columns:
        df['chol'] = df['chol'].fillna(df['chol'].median())

    # Split blood pressure
    if 'blood_pressure' in df.columns and 'systolic' not in df.columns:
        df[['systolic', 'diastolic']] = df['blood_pressure'].str.split('/', expand=True)
        df = df.drop(columns=['blood_pressure'])

    # Clean whitespace
    for col in ['systolic', 'diastolic']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Encoding
    if 'sex' in df.columns and df['sex'].dtype == object:
        df['sex'] = df['sex'].map({'female':0, 'male':1})
    if 'current_smoker' in df.columns and df['current_smoker'].dtype == object:
        df['current_smoker'] = df['current_smoker'].map({'no':0, 'yes':1})

    # Drop outlier kolestrol
    data_normal = df[df['chol'] < 400].copy()

    # Label high_bp
    data_normal["high_bp"] = (
        (data_normal["systolic"] >= 140) |
        (data_normal["diastolic"] >= 90)
    ).astype(int)

    st.write("Data setelah preparation:")
    st.write(data_normal.head())


    # 3. SPLIT DATA
    st.header("3. Train-Test Split & Scaling")

    X = data_normal.drop(['systolic', 'diastolic', 'high_bp'], axis=1)
    y = data_normal['high_bp']

    # Hapus non-numerik
    non_numeric = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=50
    )

    st.success("Model berhasil displit!")

    # Scaling
    to_scale = [c for c in X_train.columns if c not in ['sex','current_smoker']]
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[to_scale] = scaler.fit_transform(X_train[to_scale])
    X_test_scaled[to_scale] = scaler.transform(X_test[to_scale])


    # 4. TRAINING MODEL
    st.header("4. Training Random Forest")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=50
    )

    rf.fit(X_train_scaled, y_train)

    st.success("Model berhasil dilatih!")


    # 5. VISUALISASI POHON
    st.header("5. Visualisasi Beberapa Pohon Random Forest")

    n_pohon = 3
    estimators = rf.estimators_[:n_pohon]

    for i, estimator in enumerate(estimators):
        fig = plt.figure(figsize=(12, 8))
        plot_tree(
            estimator,
            feature_names=X.columns,
            class_names=["Normal", "High BP"],
            filled=True,
            fontsize=6
        )
        st.subheader(f"Pohon ke-{i+1}")
        st.pyplot(fig)


    # 6. EVALUASI MODEL
    st.header("6. Evaluasi Model")

    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)[:,1]

    st.write("**Train Accuracy:**", rf.score(X_train_scaled, y_train))
    st.write("**Test Accuracy:**", rf.score(X_test_scaled, y_test))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

with tab2:
    st.write('Visualisasi')

    # 7. Confusion Matrix
    st.subheader("7. Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)


    # 8. FEATURE IMPORTANCE
    st.header("8. Feature Importance")

    importances = rf.feature_importances_
    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fig = plt.figure(figsize=(8,5))
    sns.barplot(data=imp_df, x="importance", y="feature")
    plt.title("Feature Importance")
    st.pyplot(fig)


    # 9. ROC CURVE
    st.header("9. ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)


    # 10. PRECISION-RECALL CURVE
    st.header("10. Precision-Recall Curve")

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fig = plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    st.pyplot(fig)


    # 11. DISTRIBUSI PROBABILITAS
    st.header("11. Distribusi Probabilitas Prediksi")

    fig = plt.figure(figsize=(7,4))
    sns.histplot(y_proba[y_test==0], label="Normal", stat="density", kde=True)
    sns.histplot(y_proba[y_test==1], label="High BP", stat="density", kde=True)
    plt.legend()
    plt.title("Distribusi Probabilitas Prediksi")
    st.pyplot(fig)


    # 12. PARTIAL DEPENDENCE PLOT
    st.header("12. Partial Dependence Plot (Top 2 Fitur)")

    top_features = imp_df["feature"].tolist()[:2]

    fig, ax = plt.subplots(figsize=(8,5))
    PartialDependenceDisplay.from_estimator(rf, X_train_scaled, top_features, ax=ax)
    st.pyplot(fig)


    # 13. SCATTER TOP 2 FEATURES
    st.header("13. Scatter Plot Dua Fitur Terpenting")

    if len(top_features) >= 2:
        f1, f2 = top_features[:2]
        fig = plt.figure(figsize=(6,5))
        sns.scatterplot(x=X_test_scaled[f1], y=X_test_scaled[f2], hue=y_pred)
        plt.title(f"{f1} vs {f2}")
        st.pyplot(fig)

with tab3:
    st.write('Input Data Baru')

    # 14. PREDIKSI INPUT USER
    st.header("14. Prediksi Dengan Input Baru")

    st.subheader("Masukkan Data:")

    with st.form("pred_form"):
        age = st.number_input("Age", min_value=10, max_value=100, value=22)
        sex = st.selectbox("Sex", ["Male", "Female"])
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=82)
        chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=230)
        cigs = st.number_input("Cigarettes per Day", min_value=0, max_value=40, value=10)
        smoker = st.selectbox("Current Smoker?", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

    if submitted:

        # ---------- TEMPLATE SESUAI FITUR MODEL ----------
        input_df = pd.DataFrame(columns=X_train_scaled.columns)

        # Isi nilai sesuai input user
        input_df.loc[0, "age"] = age
        input_df.loc[0, "sex"] = 1 if sex == "Male" else 0
        input_df.loc[0, "heart_rate"] = heart_rate
        input_df.loc[0, "chol"] = chol
        input_df.loc[0, "cigs_per_day"] = cigs
        input_df.loc[0, "current_smoker"] = 1 if smoker == "Yes" else 0

        # ---------- SCALING ----------
        scaled_df = input_df.copy()
        scaled_df[to_scale] = scaler.transform(input_df[to_scale])

        # ---------- PREDIKSI ----------
        prediction = rf.predict(scaled_df)[0]
        probability = rf.predict_proba(scaled_df)[0][1]

        st.write("### Hasil Prediksi")
        st.success("Prediksi berhasil diproses.")

        hasil = "Yes" if prediction == 1 else "No"
        st.write(f"**High Blood Pressure:** {hasil}")
        st.write(f"**Probability:** `{probability:.4f}`")

        st.write("### Data yang Anda Masukkan")
        st.dataframe(input_df)