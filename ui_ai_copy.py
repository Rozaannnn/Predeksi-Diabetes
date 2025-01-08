import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    # Periksa apakah file dataset tersedia
    dataset_path = "diabetes.csv"
    if not os.path.exists(dataset_path):
        st.error(f"File '{dataset_path}' tidak ditemukan. Pastikan file ada di repository.")
        return None
    return pd.read_csv(dataset_path)

# Fungsi untuk melatih model
@st.cache_data
def train_model(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model, scaler, X_test, y_test

# Fungsi untuk prediksi data baru
def predict_new_data(model, scaler, new_data):
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)
    return prediction, probabilities

# Memuat dataset
df = load_data()

# Judul aplikasi
st.title("Prediksi Diabetes dengan Naive Bayes")
st.sidebar.header("Navigasi")

# Tab navigasi
menu = st.sidebar.selectbox("Pilih Menu", ["Deskripsi Data", "Training Model", "Prediksi Baru"])

if df is not None:  # Cek apakah dataset berhasil dimuat
    if menu == "Deskripsi Data":
        st.subheader("Deskripsi Data")
        st.write("Dataset yang digunakan untuk prediksi adalah **Pima Indians Diabetes Database**.")
        st.write("Dataset memiliki **768 baris** dan **9 kolom**, termasuk kolom target 'Outcome'.")
        st.write(df.head())
        st.write("### Statistik Deskriptif")
        st.write(df.describe())

        # Visualisasi Outcome
        st.write("### Distribusi Outcome")
        sns.set()
        fig, ax = plt.subplots()
        sns.countplot(x='Outcome', data=df, ax=ax, palette="pastel")
        ax.bar_label(ax.containers[0], label_type="center")
        st.pyplot(fig)

    elif menu == "Training Model":
        st.subheader("Training Model Naive Bayes")
        model, scaler, X_test, y_test = train_model(df)
        y_pred = model.predict(X_test)

        # Evaluasi Model
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"### Akurasi Model: {accuracy:.2f}")

        # Matriks Kebingungan
        st.write("### Matriks Kebingungan")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        st.pyplot(fig)

        # Laporan Klasifikasi
        st.write("### Laporan Klasifikasi")
        st.text(classification_report(y_test, y_pred))

    elif menu == "Prediksi Baru":
        st.subheader("Prediksi Baru")
        st.write("Masukkan data pasien untuk memprediksi kemungkinan diabetes.")

        # Input pengguna
        pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glukosa", min_value=0, max_value=200, value=85)
        blood_pressure = st.number_input("Tekanan Darah (mm Hg)", min_value=0, max_value=122, value=66)
        skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0, max_value=99, value=29)
        insulin = st.number_input("Insulin (\u03bcU/mL)", min_value=0.0, max_value=846.0, value=0.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=26.6)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
        age = st.number_input("Usia", min_value=0, max_value=120, value=31)

        if st.button("Prediksi"):
            new_data = pd.DataFrame(
                [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            )
            model, scaler, _, _ = train_model(df)
            prediction, probabilities = predict_new_data(model, scaler, new_data)

            if prediction[0] == 1:
                st.error(f"Prediksi: Pasien Mungkin Menderita Diabetes (Probabilitas: {probabilities[0][1]:.2f})")
            else:
                st.success(f"Prediksi: Pasien Tidak Menderita Diabetes (Probabilitas: {probabilities[0][0]:.2f})")

        # Informasi tambahan
        st.write("\n")
        st.info("Probabilitas menunjukkan keyakinan model terhadap prediksi.")

else:
    st.error("Dataset gagal dimuat. Harap periksa file dataset Anda.")

