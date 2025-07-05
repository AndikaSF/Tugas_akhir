import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


# Load model dan scaler
model = tf.keras.models.load_model('model_prediksi_harga_satuan_fix1.keras')
scaler_tahun = joblib.load('scaler_tahun.pkl')
scaler_harga = joblib.load('scaler_harga.pkl')

# Jenis pekerjaan 
jenis_pekerjaan_list = ['LPA', 'Pelaburan Keras', 'Burda', 'Latasir Manual']

# Judul aplikasi
st.title("Sistem Prediksi Harga Satuan dan Estimasi Biaya Proyek Jalan")

# Form input pengguna
st.subheader("Input Parameter")
tahun_input = st.number_input("Tahun", min_value=2017, max_value=2100, value=2026)
panjang = st.number_input("Panjang Jalan (meter)", min_value=0.0, value=100.0)
lebar = st.number_input("Lebar Jalan (meter)", min_value=0.0, value=3.0)
tebal = st.number_input("Ketebalan Jalan (meter)", min_value=0.0, value=0.1)
ppn_input = st.slider("PPN (%)", min_value=0, max_value=20, value=11)

# Prediksi jika tombol ditekan
if st.button("Prediksi dan Hitung Estimasi"):
    data_all = []

    # Loop jenis pekerjaan
    for pekerjaan in jenis_pekerjaan_list:
        # One-hot encoding manual
        jenis_encoding = [1 if pekerjaan == jp else 0 for jp in jenis_pekerjaan_list]

        # Gabung input
        tahun_norm = scaler_tahun.transform([[tahun_input]])[0][0]
        input_model = np.array([[tahun_norm] + jenis_encoding])

        # Prediksi dan denormalisasi
        harga_pred = model.predict(input_model)[0][0]
        harga_rupiah = scaler_harga.inverse_transform([[harga_pred]])[0][0]

        # Hitung volume
        if pekerjaan in ['Burda', 'Latasir Manual']:
            volume = panjang * lebar
            satuan = "m²"
        elif pekerjaan == 'Pelaburan Keras':
            volume = panjang * lebar * 2.5
            satuan = "ltr"
        elif pekerjaan == 'LPA':
            volume = panjang * lebar * tebal
            satuan = "m³"

        jumlah = harga_rupiah * volume

        data_all.append({
            'Jenis Pekerjaan': pekerjaan,
            'Harga Satuan': f"Rp {harga_rupiah:,.2f}",
            'Satuan': satuan,
            'Volume': f"{volume:,.2f}",
            'Jumlah Harga': jumlah
        })

    # Buat DataFrame
    df_hasil = pd.DataFrame(data_all)
    total_harga = df_hasil['Jumlah Harga'].sum()
    ppn = total_harga * (ppn_input / 100)
    total_akhir = total_harga + ppn
    dibulatkan = (np.ceil(total_akhir / 100) * 100) - 100

    # Format kolom
    df_hasil['Jumlah Harga'] = df_hasil['Jumlah Harga'].apply(lambda x: f"Rp {x:,.2f}")

    # Tampilkan tabel
    df_display = df_hasil[['Jenis Pekerjaan', 'Satuan', 'Volume', 'Harga Satuan', 'Jumlah Harga']].copy() 
    empty_row = pd.DataFrame([['', '', '', '', '']], columns=df_display.columns)
    df_display = pd.concat([df_display, empty_row], ignore_index=True)

    # Tambahkan baris total
    row_total = pd.DataFrame([['', '', '', 'Total Harga', f"Rp {total_harga:,.2f}"]], columns=df_display.columns)
    row_ppn = pd.DataFrame([['', '', '', f"PPN ({ppn_input}%)", f"Rp {ppn:,.2f}"]], columns=df_display.columns)
    row_total_akhir = pd.DataFrame([['', '', '', 'Total Setelah PPN', f"Rp {total_akhir:,.2f}"]], columns=df_display.columns)
    row_bulat = pd.DataFrame([['', '', '', 'Dibulatkan', f"Rp {dibulatkan:,.2f}"]], columns=df_display.columns)

    # Gabungkan semua
    df_display = pd.concat([df_display, row_total, row_ppn, row_total_akhir, row_bulat], ignore_index=True)

    # Tampilkan hasil akhir
    st.subheader("Hasil Estimasi dan Perhitungan Total")
    st.dataframe(df_display)

    # Visualisasi Bar Chart Harga Satuan
    st.subheader("Visualisasi Harga Satuan")
    harga_numerik = [float(h.replace("Rp ", "").replace(",", "")) for h in df_hasil['Harga Satuan']]
    fig_bar, ax_bar = plt.subplots()
    sns.barplot(x=df_hasil['Jenis Pekerjaan'], y=harga_numerik, ax=ax_bar, palette="Blues_d")
    ax_bar.set_ylabel("Harga Satuan (Rp)")
    ax_bar.set_xlabel("Jenis Pekerjaan")
    ax_bar.set_title("Harga Satuan per Jenis Pekerjaan")
    st.pyplot(fig_bar)

    # Visualisasi Pie Chart Proporsi Biaya
    st.subheader("Proporsi Estimasi Biaya per Jenis Pekerjaan")
    jumlah_numerik = [float(j.replace("Rp ", "").replace(",", "")) for j in df_hasil['Jumlah Harga']]
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(jumlah_numerik, labels=df_hasil['Jenis Pekerjaan'], autopct='%1.1f%%', startangle=140)
    ax_pie.axis('equal')  # Membuat pie chart berbentuk lingkaran
    st.pyplot(fig_pie)
