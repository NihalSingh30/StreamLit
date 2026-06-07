import streamlit as st
import numpy as np
import pandas as pd
import sys, os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Sleep & GPA Predictor",
    page_icon="🌙",
    layout="centered",
)

# ── Load / train model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model prediksi...")
def get_model():
    from train import load_or_train
    return load_or_train()

model, meta = get_model()

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_features(sleep_hours, sleep_start_hour, irregularity, nap_min, cum_gpa, gender, firstgen):
    total_sleep_min  = sleep_hours * 60
    sleep_start_min  = sleep_start_hour * 60
    midpoint_min     = (sleep_start_min + total_sleep_min / 2) % (24 * 60)
    irregularity_mssd = (irregularity - 1) / 4 * 2.0 + 0.05
    return np.array([[
        total_sleep_min, midpoint_min, irregularity_mssd,
        nap_min, 1.0, cum_gpa, gender, firstgen
    ]])

def gpa_label(gpa):
    if gpa >= 3.7:  return "Sangat Baik", "#1a7a4a"
    if gpa >= 3.3:  return "Baik", "#2d8a3e"
    if gpa >= 2.7:  return "Cukup", "#b07d00"
    if gpa >= 2.0:  return "Di Bawah Rata-rata", "#c45000"
    return "Kritis", "#c0392b"

def generate_recommendations(sleep_hours, sleep_start_hour, irregularity, nap_min, predicted_gpa, cum_gpa):
    recs = []

    if sleep_hours < 7:
        deficit = 7 - sleep_hours
        recs.append(("🛌 Durasi Tidur", 
            f"Kamu tidur rata-rata {sleep_hours:.1f} jam/malam, kurang dari rekomendasi minimum 7 jam. "
            f"Tambah {deficit:.1f} jam tidur bisa signifikan memengaruhi konsolidasi memori dan performa akademik."))

    if sleep_hours > 9:
        recs.append(("🛌 Durasi Tidur",
            f"Tidur {sleep_hours:.1f} jam/malam terlalu banyak untuk sebagian orang dan bisa menandakan kualitas tidur yang buruk atau depresi. "
            "Kalau kamu tetap merasa lelah, konsultasi ke dokter lebih berguna daripada tidur lebih lama."))

    norm_start = sleep_start_hour if sleep_start_hour <= 24 else sleep_start_hour - 24
    if norm_start >= 2 and norm_start <= 12:
        recs.append(("⏰ Waktu Tidur",
            f"Jam tidur {sleep_start_hour:.0f}:00 termasuk sangat larut. "
            "Data menunjukkan mahasiswa yang tidur sebelum jam 01:00 cenderung punya performa akademik lebih baik. "
            "Geser waktu tidur 30 menit lebih awal setiap minggu lebih efektif daripada langsung berubah drastis."))
    elif sleep_start_hour >= 24 or (sleep_start_hour >= 0 and sleep_start_hour < 2):
        recs.append(("⏰ Waktu Tidur",
            "Tidur di atas jam 00:00 berkorelasi dengan peningkatan fragmentasi tidur dan penurunan slow-wave sleep. "
            "Target idealnya tidur sebelum jam 23:30."))

    if irregularity >= 3.5:
        recs.append(("📅 Konsistensi Jadwal",
            f"Keteraturan jadwal tidur kamu di level {irregularity:.1f}/5, yang tergolong tinggi. "
            "Irregularitas jadwal tidur adalah prediktor performa akademik yang lebih kuat dari durasi tidur itu sendiri di dataset ini. "
            "Coba tetapkan alarm tidur (bukan hanya alarm bangun) dan pertahankan jadwal yang sama di akhir pekan."))
    elif irregularity >= 2.5:
        recs.append(("📅 Konsistensi Jadwal",
            "Jadwal tidur kamu cukup tidak teratur. Variasi >1 jam antara hari kerja dan akhir pekan sudah cukup mengganggu ritme sirkadian."))

    if nap_min > 90:
        recs.append(("💤 Tidur Siang",
            f"Tidur siang {nap_min:.0f} menit/hari lebih dari cukup dan bisa mengurangi sleep pressure malam hari sehingga jam tidur malam makin mundur. "
            "Batasi tidur siang maksimal 20-30 menit dan hindari setelah jam 15:00."))
    elif nap_min == 0 and sleep_hours < 6.5:
        recs.append(("💤 Tidur Siang",
            "Dengan total tidur malam yang kurang, tidur siang singkat 20 menit bisa membantu fungsi kognitif tanpa mengganggu tidur malam."))

    if predicted_gpa < cum_gpa - 0.3:
        recs.append(("📚 Performa Akademik",
            f"Prediksi IPK semester ini ({predicted_gpa:.2f}) lebih rendah {cum_gpa - predicted_gpa:.2f} poin dari IPK kumulatif kamu ({cum_gpa:.2f}). "
            "Perubahan pola tidur adalah kontributor yang bisa kamu kendalikan sekarang, tapi perlu diimbangi dengan evaluasi beban studi dan strategi belajar."))

    if not recs:
        recs.append(("✅ Pola Tidur",
            "Pola tidur kamu sudah cukup baik berdasarkan data yang diinput. "
            "Pertahankan konsistensi jadwal dan pastikan kualitas tidur (gelap, dingin, bebas layar 30 menit sebelum tidur)."))

    return recs

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("## 🌙 Sleep & GPA Predictor")
st.markdown(
    "Prediksi IPK semester berdasarkan pola tidur. "
    "Model dilatih dari data mahasiswa Carnegie Mellon University & University of Washington "
    f"(n={meta['n_samples']})."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Pola Tidur**")
    sleep_hours = st.slider("Rata-rata jam tidur per malam", 2.0, 12.0, 7.0, 0.5,
                            help="Estimasi rata-rata selama semester berjalan")
    sleep_start = st.slider("Jam mulai tidur", 20.0, 30.0, 23.0, 0.5,
                            help="20 = jam 20:00, 25 = jam 01:00 dini hari, 26 = jam 02:00, dst.")
    st.caption(f"Waktu tidur: {'%02d:%02d' % (int(sleep_start) % 24, int((sleep_start % 1) * 60))}")

    irregularity = st.slider("Konsistensi jadwal tidur", 1.0, 5.0, 2.0, 0.5,
                             help="1 = tidur jam yang sama setiap hari, 5 = sangat tidak menentu")
    nap_min = st.number_input("Rata-rata tidur siang (menit/hari)", 0, 300, 15, 5)

with col2:
    st.markdown("**Akademik & Demografi**")
    cum_gpa = st.number_input("IPK kumulatif sebelumnya (0.0 – 4.0)", 0.0, 4.0, 3.2, 0.01)
    gender = st.selectbox("Gender", ["Perempuan", "Laki-laki"])
    firstgen = st.selectbox("First-generation student?", ["Tidak", "Ya"])

gender_val   = 1.0 if gender == "Perempuan" else 0.0
firstgen_val = 1.0 if firstgen == "Ya" else 0.0

st.divider()

if st.button("Prediksi IPK", type="primary", use_container_width=True):
    X = build_features(sleep_hours, sleep_start, irregularity, nap_min, cum_gpa, gender_val, firstgen_val)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = float(np.clip(model.predict(X)[0], 0.0, 4.0))

    label, color = gpa_label(pred)

    st.markdown("### Hasil Prediksi")

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediksi IPK Semester", f"{pred:.2f}")
    m2.metric("IPK Kumulatif Sebelumnya", f"{cum_gpa:.2f}", f"{pred - cum_gpa:+.2f}")
    m3.metric("Status", label)

    mae = meta['mae']
    st.caption(f"Margin error model: ±{mae:.2f} GPA points (MAE) &nbsp;|&nbsp; R² = {meta['r2']:.2f}")

    st.progress(min(pred / 4.0, 1.0))

    st.markdown("### Rekomendasi")
    recs = generate_recommendations(sleep_hours, sleep_start, irregularity, nap_min, pred, cum_gpa)
    for title, body in recs:
        with st.expander(title, expanded=True):
            st.write(body)

    st.divider()
    st.caption(
        "Model: Gradient Boosting Regressor | Fitur: durasi tidur, waktu tidur, irregularitas jadwal, "
        "tidur siang, IPK kumulatif, gender, first-gen status. "
        "Prediksi ini adalah estimasi statistik, bukan jaminan. "
        "Banyak faktor lain di luar pola tidur yang memengaruhi IPK."
    )
