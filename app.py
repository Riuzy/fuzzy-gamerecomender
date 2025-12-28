import streamlit as st
import pandas as pd
import numpy as np
import time 

# Asumsi kurs: 1 USD = Rp 15.500 
IDR_RATE = 15500 

from src.data_prep import clean_and_preprocess
from src.system_fuzzy import dapatkan_simulator_fuzzy, hitung_skor_fuzzy

try:
    simulator_fuzzy = dapatkan_simulator_fuzzy()
except Exception as e:
    simulator_fuzzy = None
    
# --- 1. PEMUATAN DATA DENGAN CACHING STREAMLIT ---
@st.cache_data
def load_data_from_processed_file():
    """Memuat data yang sudah diproses dari CSV hanya sekali."""
    DATA_PATH = './data/processed_data.csv'
    JSON_PATH = './data/games.json'
    
    try:
        df = pd.read_csv(DATA_PATH)
        required_scores = ['Quality_Score', 'Popularity_Score', 'Price_Score']
        if not all(col in df.columns for col in required_scores):
             st.error("Kolom skor Fuzzy tidak ditemukan. Lakukan pemrosesan ulang data.")
             return pd.DataFrame()
        
        for col in required_scores:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=required_scores, inplace=True)
        df.sort_values(by='release_year', inplace=True)
        
        return df
    except FileNotFoundError:
        st.error("File processed_data.csv tidak ditemukan. Mencoba memproses data mentah...")
        df = clean_and_preprocess(JSON_PATH)
        if df.empty:
            st.error("Gagal memproses data. Cek games.json.")
            return pd.DataFrame()
        return df

def apply_genre_filter(df, liked_genres):
    """Filter DataFrame berdasarkan kesamaan genre."""
    if not liked_genres:
        return df.copy()

    def genre_filter_match(genres_str):
        if pd.isna(genres_str):
            return False
        game_genres = set(g.strip().lower() for g in str(genres_str).split(';') if g.strip()) 
        return bool(game_genres.intersection(set(liked_genres)))

    filtered_df = df[df['genres'].apply(genre_filter_match)].copy()
    return filtered_df

# --- 2. FUNGSI PERHITUNGAN FUZZY ---
@st.cache_data
def calculate_all_fuzzy_scores(df_filtered):
    """Menghitung skor Fuzzy untuk semua game yang difilter (Menggunakan loop NumPy)."""
    
    if simulator_fuzzy is None:
        return df_filtered.assign(Fuzzy_Score=0)

    input_data = df_filtered[['Quality_Score', 'Popularity_Score', 'Price_Score']].values
    fuzzy_scores = []
    
    for row in input_data:
        q_score, p_score, pr_score = row
        score = hitung_skor_fuzzy(q_score, p_score, pr_score, simulator_fuzzy)
        fuzzy_scores.append(score)
        
    df_filtered['Fuzzy_Score'] = fuzzy_scores
    return df_filtered

# --- 3. ANTARMUKA STREAMLIT UTAMA ---
st.set_page_config(page_title="Steam Fuzzy Recommender", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'> Sistem Rekomendasi Game Steam (Fuzzy Logic)</h1>", 
    unsafe_allow_html=True
)

df_global = load_data_from_processed_file()
if df_global.empty or simulator_fuzzy is None:
    st.stop()
    
max_price_usd = df_global['price_original'].max()
max_price_idr = max_price_usd * IDR_RATE

min_year = int(df_global['release_year'].min())
max_year = int(df_global['release_year'].max())

if min_year == 0 or max_year == 0:
    min_year = 2000
    max_year = pd.Timestamp.now().year

# --- TATA LETAK UTAMA ---
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    st.header("Input Preferensi Anda")

    with st.form("recommender_form"):
        
        genre_input = st.text_input(
            "Genre yang Disukai (Pisahkan dengan koma):", 
            value="Casual, Indie, Sports", 
            placeholder="cth: Action, RPG, Strategy"
        )
        
        # INPUT HARGA DALAM RUPIAH
        budget_input_idr = st.number_input(
            f"Anggaran Maksimal (Rupiah): (Maks: Rp {max_price_idr:,.0f})", 
            min_value=0.0, 
            max_value=max_price_idr,
            value=20000.00,
            step=5000.0
        )

        # WIDGET FILTER TAHUN RILIS
        year_range = st.slider(
            "Rentang Tahun Rilis:",
            min_value=min_year,
            max_value=max_year,
            value=(max_year - 5, max_year), 
            step=1
        )
        st.caption(f"Menyaring game dari tahun **{year_range[0]}** hingga **{year_range[1]}**.")
        
        submitted = st.form_submit_button("Cari Rekomendasi", type="primary")

    if submitted:
        
        start_time = time.time()
        
        # Konversi budget IDR ke USD untuk filtering (karena data price_original adalah USD)
        budget_usd = budget_input_idr / IDR_RATE
        
        # --- 1. Filter Data Berdasarkan Input Pengguna ---
        # A. Filter Genre
        liked_genres = [g.strip().lower() for g in genre_input.split(',') if g.strip()]
        filtered_by_genre = apply_genre_filter(df_global, liked_genres)

        # B. Filter Harga (menggunakan USD)
        filtered_by_price = filtered_by_genre[filtered_by_genre['price_original'] <= budget_usd].copy()
        
        # C. Filter Tahun Rilis
        filtered_data = filtered_by_price[
            (filtered_by_price['release_year'] >= year_range[0]) & 
            (filtered_by_price['release_year'] <= year_range[1])
        ].copy()
        
        st.info(f"Game yang lolos 3 filter (Genre, Harga, Tahun): **{len(filtered_data)}** game.")
        
        if filtered_data.empty:
            st.warning("Tidak ada game yang cocok dengan semua kriteria Anda.")
            st.stop()

        # --- 2. Perhitungan Fuzzy ---
        st.subheader(f"Mulai menghitung Fuzzy Score pada {len(filtered_data)} game...")
        
        filtered_data.dropna(subset=['Quality_Score', 'Popularity_Score', 'Price_Score'], inplace=True)
        
        df_with_score = calculate_all_fuzzy_scores(filtered_data)
        
        # --- 3. Tampilkan Hasil ---
        final_results = df_with_score.sort_values(by='Fuzzy_Score', ascending=False)
        top_recommendations = final_results.head(20) 
        
        end_time = time.time()
        st.success(f"✅ Komputasi Selesai dalam **{end_time - start_time:.2f} detik**.")
        
        st.header(" 20   Rekomendasi Game Terbaik")
        
        display_df = top_recommendations.copy()
        
        # Konversi Harga ke Rupiah
        display_df['Harga (IDR)'] = (display_df['price_original'] * IDR_RATE).apply(
            lambda x: f"Rp {x:,.0f}"
        )
        display_df['Harga ($)'] = display_df['price_original'].apply(
            lambda x: f"${x:.2f}"
        )

        final_display_df = display_df[[
            'Fuzzy_Score',
            'name',
            'genres',
            'categories',
            'release_date_str',
            'Harga (IDR)',
            'Harga ($)'
        ]]
        final_display_df.columns = [
            'Skor Fuzzy',
            'Nama Game',
            'Genre',
            'Kategori',
            'Tanggal Rilis',
            'Harga (Rupiah)',
            'Harga (USD)'
        ]
        
        st.dataframe(
            final_display_df, 
            hide_index=True, 
            use_container_width=True
        )