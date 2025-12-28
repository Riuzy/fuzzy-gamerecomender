import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- 1. DEFINISI VARIABEL LINGUISTIK ---
kualitas = ctrl.Antecedent(np.arange(0, 101, 1), 'kualitas')
popularitas = ctrl.Antecedent(np.arange(0, 101, 1), 'popularitas')
harga = ctrl.Antecedent(np.arange(0, 101, 1), 'harga') 
rekomendasi = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')

# --- 2. PEMBUATAN HIMPUNAN FUZZY ---

# kualitas
kualitas['rendah'] = fuzz.trimf(kualitas.universe, [0, 0, 50])
kualitas['sedang'] = fuzz.trimf(kualitas.universe, [30, 65, 85])
kualitas['tinggi'] = fuzz.trimf(kualitas.universe, [70, 100, 100])

# popuralitas
popularitas['rendah'] = fuzz.trimf(popularitas.universe, [0, 0, 30])
popularitas['sedang'] = fuzz.trimf(popularitas.universe, [15, 40, 75])
popularitas['tinggi'] = fuzz.trimf(popularitas.universe, [50, 100, 100])

# harga
harga['mahal'] = fuzz.trimf(harga.universe, [0, 0, 35])
harga['normal'] = fuzz.trimf(harga.universe, [20, 50, 80])
harga['murah'] = fuzz.trimf(harga.universe, [65, 100, 100])

# hasil rekomendasi
rekomendasi['lemah'] = fuzz.trimf(rekomendasi.universe, [0, 0, 40])
rekomendasi['sedang'] = fuzz.trimf(rekomendasi.universe, [30, 60, 80])
rekomendasi['kuat'] = fuzz.trimf(rekomendasi.universe, [70, 100, 100])

# --- 3. PERUMUSAN ATURAN FUZZY (RULE BASE) ---
rule1 = ctrl.Rule(kualitas['tinggi'] & popularitas['tinggi'] & harga['murah'], rekomendasi['kuat'])
rule2 = ctrl.Rule(kualitas['tinggi'] & popularitas['sedang'] & harga['murah'], rekomendasi['kuat'])
rule3 = ctrl.Rule(kualitas['tinggi'] & popularitas['tinggi'] & harga['mahal'], rekomendasi['sedang'])
rule4 = ctrl.Rule(kualitas['sedang'] & popularitas['tinggi'] & harga['murah'], rekomendasi['kuat'])
rule5 = ctrl.Rule(kualitas['rendah'], rekomendasi['lemah'])
rule6 = ctrl.Rule(kualitas['sedang'] & popularitas['sedang'] & harga['normal'], rekomendasi['sedang'])
rule7 = ctrl.Rule(popularitas['rendah'] & harga['mahal'], rekomendasi['lemah'])
rule8 = ctrl.Rule(kualitas['tinggi'] & popularitas['rendah'] & harga['murah'], rekomendasi['sedang'])


# --- 4. PEMBUATAN SISTEM KONTROL DAN SIMULATOR ---
rekomendasi_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])

def dapatkan_simulator_fuzzy():
    return ctrl.ControlSystemSimulation(rekomendasi_ctrl)

def hitung_skor_fuzzy(skor_kualitas, skor_popularitas, skor_harga, simulator):
    try:
        simulator.input['kualitas'] = skor_kualitas
        simulator.input['popularitas'] = skor_popularitas
        simulator.input['harga'] = skor_harga
        simulator.compute()
        return simulator.output['rekomendasi']
    except (ValueError, KeyError):
        return 0.0