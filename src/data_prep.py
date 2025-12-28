import pandas as pd
import numpy as np
import json
import re

def clean_and_preprocess(file_path):
    
    # --- 1. MEMBACA DATASET DARI FILE JSON ---
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        records = []
        for appid, details in data.items():
            record = {'appid': appid, 'name': details.get('name')}
            
            # Ekstraksi Kolom yang Diperlukan
            record['price_original'] = details.get('price', 0)
            record['release_date_str'] = details.get('release_date', 'N/A')
            record['positive_ratings'] = details.get('positive', 0)
            record['negative_ratings'] = details.get('negative', 0)
            record['estimated_owners_str'] = details.get('estimated_owners', '0 - 0')
            record['peak_ccu'] = details.get('peak_ccu', 0)
            
            # List/Array
            record['genres'] = ';'.join(details.get('genres', []))
            record['categories'] = ';'.join(details.get('categories', []))
            
            # Tags (untuk popularitas/tampilan)
            tags_data = details.get('tags', [])
            if isinstance(tags_data, dict):
                record['tags'] = ';'.join(tags_data.keys())
            else:
                record['tags'] = ''

            records.append(record)
            
        df = pd.DataFrame(records)
        
    except Exception as e:
        print(f"GAGAL KRITIS: Gagal memproses file JSON. Detail: {e}")
        return pd.DataFrame()

    if df.empty:
        print(" Data frame kosong setelah pembacaan.")
        return pd.DataFrame()

    # --- 2. CLEANING DAN KONVERSI TIPE DATA ---
    df.dropna(subset=['name'], inplace=True)

    # Cleaning Release Date (untuk filter tahun)
    def extract_year(date_str):
        try:
            return pd.to_datetime(date_str, errors='coerce').year
        except:
            return np.nan
            
    df['release_year'] = df['release_date_str'].apply(extract_year)
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(0).astype(int)

    # Ekstraksi Estimated Owners (Mengambil rata-rata)
    def clean_owners(owner_str):
        match = re.findall(r'\d+', str(owner_str).replace(',', ''))
        if len(match) == 2:
            low, high = int(match[0]), int(match[1])
            return (low + high) / 2
        return 0
    df['estimated_owners'] = df['estimated_owners_str'].apply(clean_owners)
    
    # Konversi numerik
    df['price_original'] = pd.to_numeric(df['price_original'], errors='coerce').fillna(0)
    df['positive_ratings'] = pd.to_numeric(df['positive_ratings'], errors='coerce').fillna(0)
    df['negative_ratings'] = pd.to_numeric(df['negative_ratings'], errors='coerce').fillna(0)
    df['peak_ccu'] = pd.to_numeric(df['peak_ccu'], errors='coerce').fillna(0)

    # --- 3. PERHITUNGAN SCORE (FUZZY INPUT) ---
    
    # Kualitas (0-100)
    df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
    df['Quality_Score'] = np.where(df['total_ratings'] > 0, (df['positive_ratings'] / df['total_ratings']) * 100, 0)

    # Popularitas (0-100)
    df['Popularity_Raw'] = np.log1p(df['estimated_owners']) * 0.7 + np.log1p(df['peak_ccu']) * 0.3
    min_pop = df['Popularity_Raw'].min()
    max_pop = df['Popularity_Raw'].max()
    epsilon = 1e-9
    range_pop = max_pop - min_pop
    
    if range_pop > epsilon:
        df['Popularity_Score'] = ((df['Popularity_Raw'] - min_pop) / range_pop) * 100
    else:
        df['Popularity_Score'] = 0 

    # Harga (Price_Score: 0-100)
    min_price = df['price_original'].min()
    max_price = df['price_original'].max()
    
    if max_price > epsilon:
        df['Price_Score'] = (1 - (df['price_original'] / max_price)) * 100
    else:
        df['Price_Score'] = 100 

    # --- 4. MENYIMPAN DATA YANG DIPROSES ---
    
    # Kolom Final yang Akan Disediakan untuk Streamlit (menggunakan 'appid' sebagai ID)
    processed_df = df[[
        'appid', 'name', 'genres', 'categories', 'release_date_str', 
        'release_year', 'price_original', 'positive_ratings', 'negative_ratings', 
        'Quality_Score', 'Popularity_Score', 'Price_Score'
    ]].copy()
    
    processed_df.to_csv('./data/processed_data.csv', index=False)
    
    print("\n✅ Data pre-processing selesai. Kolom id_game telah dihapus.")
    return processed_df

if __name__ == '__main__':
    clean_and_preprocess('./data/games.json')