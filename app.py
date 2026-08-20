import os
import pickle
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


# ==============================================================================
# PROSES PEMUATAN ARTIFAK MODEL DARI GITHUB RELEASES
# ==============================================================================
@st.cache_resource
def load_multimodal_artifacts():
    os.makedirs("saved_models", exist_ok=True)

    MODEL_PATH = "saved_models/vit_tabular_coffee_model.pth"
    SCALER_PATH = "saved_models/tabular_scaler.pkl"
    LABEL_ENCODER_PATH = "saved_models/label_encoder.pkl"

    # GANTI URL DI BAWAH INI DENGAN LINK RELEASES GITHUB ANDA DARI LANGKAH SEBELUMNYA
    FILE_URLS = {
        MODEL_PATH: "https://github.com/Riduan-Syahri/sistem-pakar-kopi-multimodal/releases/download/v1.0.0/vit_tabular_coffee_model.pth",
        SCALER_PATH: "https://github.com/Riduan-Syahri/sistem-pakar-kopi-multimodal/releases/download/v1.0.0/tabular_scaler.pkl",
        LABEL_ENCODER_PATH: "https://github.com/Riduan-Syahri/sistem-pakar-kopi-multimodal/releases/download/v1.0.0/label_encoder.pkl",
    }

    # Auto-Download file dari GitHub Releases menggunakan urllib (Bebas error HTML Google Drive)
    for file_path, download_url in FILE_URLS.items():
        if not os.path.exists(file_path):
            with st.spinner(
                f"Mengunduh berkas {os.path.basename(file_path)}..."
            ):
                urllib.request.urlretrieve(download_url, file_path)

    image_processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224"
    )

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    num_classes = len(le.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTTabularFusionModel(num_classes=num_classes, tabular_dim=3)
    state_dict = torch.load(
        MODEL_PATH, map_location=device, weights_only=False
    )

    # Logika Auto-Mapping Lintas Versi Hugging Face / PyTorch
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if "vit.encoder.layer." in key:
            new_key = key.replace("vit.encoder.layer.", "vit.layers.")
            new_key = new_key.replace(
                ".attention.attention.query.", ".attention.q_proj."
            )
            new_key = new_key.replace(
                ".attention.attention.key.", ".attention.k_proj."
            )
            new_key = new_key.replace(
                ".attention.attention.value.", ".attention.v_proj."
            )
            new_key = new_key.replace(
                ".attention.output.dense.", ".attention.o_proj."
            )
            new_key = new_key.replace(".intermediate.dense.", ".mlp.fc1.")
            new_key = new_key.replace(".output.dense.", ".mlp.fc2.")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, image_processor, scaler, le, device

# ==============================================================================
# 3. BASIS PENGETAHUAN (KNOWLEDGE BASE) SOLUSI KOMPREHENSIF PENYAKIT KOPI
# ==============================================================================
SOLUSI_PENYAKIT = {
    "karat daun": {
        "nama_latin": "Hemileia vastatrix",
        "deskripsi": "Penyakit utama tanaman kopi. Gejala berupa bercak kuning muda di sisi bawah daun yang lama-kelamaan berubah menjadi jingga terang seperti serbuk bedak akibat tumpukan spora jamur.",
        "pencegahan": [
            "Gunakan varietas kopi tahan karat daun (seperti rumpun Andungsari 1, Sigarar Utang, atau Catimor).",
            "Atur pangkasan pohon pelindung kebun secara berkala agar masuknya sinar matahari optimal dan menurunkan kelembapan mikro kebun.",
            "Semprotkan agens hayati jamur kompetitor alami seperti *Trichoderma harzianum* sebelum memasuki musim hujan."
        ],
        "perbaikan": [
            "Segera potong daun atau cabang yang terinfeksi parah, kumpulkan di luar area kebun lalu bakar agar spora tidak diterbangkan angin ke pohon sehat.",
            "Berikan pupuk tambahan yang kaya unsur Kalium (K) seperti KCl atau NPK berkalium tinggi untuk mempertebal lapisan sel epidermis daun."
        ],
        "obat_kimia": "Semprotkan Fungisida Kontak berbahan aktif Tembaga (contoh: Nordox 56 WP, Cupravit, atau Vitigran Blue) dengan dosis 2 gram/liter air, atau Fungisida Sistemik golongan Triazole (contoh: Anvil 50 SC atau Score 250 EC) jika serangan melebihi 15% populasi."
    },
    "bercak daun": {
        "nama_latin": "Cercospora coffeicola",
        "deskripsi": "Disebabkan oleh infeksi jamur Cercospora. Gejala berupa bercak bulat cokelat tua dengan pusat berwarna abu-abu cerah seperti mata burung (brown eye spot). Sering timbul akibat tanaman kekurangan hara.",
        "pencegahan": [
            "Pastikan jarak tanam di kebun produksi ideal (tidak terlalu rapat) untuk meminimalisir penularan gesekan daun.",
            "Jaga kebersihan kebun dari gulma atau tanaman liar pelindung mikro yang bisa menjadi inang alternatif jamur patogen."
        ],
        "perbaikan": [
            "Lakukan pemupukan berimbang, terutama pemulihan darurat unsur Nitrogen (N) menggunakan pupuk Urea atau pupuk daun komersial (seperti Gandasil D) guna memicu pertumbuhan daun baru.",
            "Rontokkan daun bergejala awal yang berada di ranting bagian bawah kebun."
        ],
        "obat_kimia": "Semprotkan Fungisida Kontak berbahan aktif Mankozeb (contoh: Dithane M-45 atau Antracol 70 WP) or fungisida sistemik berbahan aktif Karbendazim / Azoksistrobin dengan interval 10-14 hari sekali pada musim hujan."
    },
    "hawar daun": {
        "nama_latin": "Rhizoctonia solani / Pellicularia koleroga",
        "deskripsi": "Daun kopi mendadak tampak layu, kering cokelat tua kehitaman seperti tersiram air panas dimulai dari bagian ujung. Daun kering biasanya tetap menggantung di ranting karena terikat oleh jalinan benang jamur tipis.",
        "pencegahan": [
            "Perbaiki sistem drainase/parit kebun agar air tanah tidak menggenang, karena penyakit ini sangat menyukai kelembapan tanah di atas 85%.",
            "Lakukan pemangkasan bentuk pada pohon kopi untuk memaksimalkan aerasi sirkulasi udara di dalam tajuk pohon."
        ],
        "perbaikan": [
            "Potong ranting yang daunnya membusuk hingga 5 cm ke arah kayu yang masih sehat.",
            "Wajib mensterilkan alat gunting pangkas menggunakan alkohol 70% setelah memotong bagian sakit agar tidak menularkan patogen secara mekanis ke pohon sehat lainnya."
        ],
        "obat_kimia": "Semprotkan Fungisida Berbahan Aktif Tembaga Hidroksida (contoh: Funguran atau Kocide 54 WD) atau fungisida sistemik berbahan aktif Difenokonazol yang ditargetkan langsung pada area batang dan ranting yang terinfeksi."
    },
    "embun jelaga": {
        "nama_latin": "Capnodium coffeae",
        "deskripsi": "Terbentuknya lapisan kerak tipis berwarna hitam pekat menyerupai jelaga lampu di permukaan atas daun kopi. Jamur ini sekadar menumpang hidup pada kotoran manis (sekresi madu) yang dikeluarkan oleh hama kutu daun.",
        "pencegahan": [
            "Fokus utama adalah mengendalikan koloni hama pencetusnya (kutu dompolan, kutu sisik, atau kutu putih). Jika hama kutu habis, jamur ini tidak akan mampu berkembang.",
            "Bersihkan sarang semut di sekitar pangkal pohon yang sering menjadi pelindung mobilitas kutu tanaman."
        ],
        "perbaikan": [
            "Pangkas ranting yang tertutup jelaga sangat tebal karena lapisan hitam tersebut menghalangi proses fotosintesis secara total.",
            "Gunakan semprotan air bersih bertekanan sedang yang dicampur sedikit sabun antiseptik pertanian organik untuk meluruhkan lapisan kerak hitam pada daun."
        ],
        "obat_kimia": "Gunakan Insektisida sistemik untuk membasmi kutu daun berbahan aktif Imidakloprid (contoh: Confidor 200 SL atau Winder 100 EC), lalu padukan dengan Fungisida Kontak berbahan aktif Tembaga (contoh: Nordox 56 WP) dosis 2 gram/liter air."
    },
    "sehat": {
        "nama_latin": "Healthy Plant (Kondisi Optimal)",
        "deskripsi": "Jaringan klorofil daun tanaman kopi berada dalam keadaan prima, metabolisme sel berjalan seimbang, dan bebas dari kolonisasi mikroorganisme patogen.",
        "pencegahan": [
            "Lanjutkan pemeliharaan berkala melalui pemupukan organik dengan kompos atau pupuk kandang matang sebanyak 10-15 kg per pohon setahun sekali.",
            "Lakukan monitoring kestabilan derajat keasaman tanah. Jika pH berada di bawah 5.5, taburkan kapur pertanian (Dolomit) untuk menaikkan pH ke angka optimal (5.5 - 6.5)."
        ],
        "perbaikan": [
            "Tidak memerlukan tindakan perbaikan atau pemotongan jaringan.",
            "Dapat diaplikasikan pupuk mikro (mengandung Fe, Zn, B) atau pupuk NPK seimbang (15-15-15) secara berkala guna mempertahankan stabilitas sistem imun alami pohon."
        ],
        "obat_kimia": "Tanaman sehat tidak membutuhkan racun kimia (Fungisida/Insektisida). Penggunaan racun pada pohon sehat justru berisiko mematikan mikroorganisme endofit yang menguntungkan bagi daun tanaman."
    }
}


# ==============================================================================
# 4. MANAJEMEN HALAMAN UTAMA (STREAMLIT UI/UX MULTI-PAGE)
# ==============================================================================
st.set_page_config(page_title="Sistem Pakar Multimodal Kopi", layout="centered")

if 'halaman_aktif' not in st.session_state:
    st.session_state.halaman_aktif = 'welcome'

try:
    model, image_processor, scaler, le, device = load_multimodal_artifacts()
except Exception as e:
    st.error(f"Gagal memuat komponen sistem: {e}")
    st.stop()


# ------------------------------------------------------------------------------
# KONDISI A: HALAMAN SELAMAT DATANG (WELCOME SCREEN)
# ------------------------------------------------------------------------------
if st.session_state.halaman_aktif == 'welcome':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?q=80&w=600&auto=format&fit=crop", caption="Smart Agriculture Coffee Research", use_container_width=True)
    
    st.title("🌱 Selamat Datang di Sistem Pakar Diagnosa Kopi")
    st.subheader("Platform Multimodal Deep Learning Berbasis Vision Transformer & MLP")
    
    st.markdown("""
    Aplikasi cerdas ini dirancang khusus untuk mendeteksi kesehatan tanaman kopi secara akurat dengan menggabungkan dua dimensi data:
    1. **Data Visual:** Foto kerusakan fisik pada permukaan daun kopi.
    2. **Data Agro-Klimat:** Parameter sensor lingkungan kebun (Kelembapan Udara, Kelembapan Tanah, dan pH).
    
    Sistem akan melakukan ekstraksi fusi fitur secara *real-time* guna memberikan hasil diagnosa serta rekomendasi solusi penanganan yang komprehensif.
    """)
    
    st.markdown("---")
    if st.button("🚀 Mulai Proses Diagnosa", type="primary", use_container_width=True):
        st.session_state.halaman_aktif = 'sistem_pakar'
        st.rerun()


# ------------------------------------------------------------------------------
# KONDISI B: HALAMAN UTAMA INTERAKSI SISTEM PAKAR
# ------------------------------------------------------------------------------
elif st.session_state.halaman_aktif == 'sistem_pakar':
    if st.button("⬅️ Kembali ke Menu Awal"):
        st.session_state.halaman_aktif = 'welcome'
        st.rerun()
        
    st.title("🖥️ Panel Diagnosa Multimodal")
    st.write("Silakan lengkapi parameter sensor lingkungan dan pilih metode input foto sampel daun di bawah ini.")
    st.markdown("---")
    
    # Form Input 1: Parameter Tabular (Sensor)
    st.subheader("📋 1. Input Parameter Agro-Klimat Kebun")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_udara = st.number_input("Kelembapan Udara (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    with col2:
        input_tanah = st.number_input("Kelembapan Tanah (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.1)
    with col3:
        input_ph = st.number_input("pH Tanah", min_value=0.0, max_value=14.0, value=6.0, step=0.1)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Form Input 2: Metode Masukan Gambar Daun (Pilihan: File Upload atau Kamera)
    st.subheader("📸 2. Ambil / Unggah Foto Gejala Kerusakan Daun")
    
    # Membuat Tab Menu agar tampilan tetap bersih dan elegan
    tab_upload, tab_kamera = st.tabs(["📁 Unggah Berkas Foto", "📷 Ambil Foto lewat Kamera"])
    
    final_image = None
    
    with tab_upload:
        uploaded_file = st.file_uploader("Pilih file citra daun (format: JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"], key="file_key")
        if uploaded_file is not None:
            final_image = Image.open(uploaded_file).convert("RGB")
            
    with tab_kamera:
        camera_file = st.camera_input("Arahkan kamera laptop/HP langsung ke permukaan daun kopi", key="camera_key")
        if camera_file is not None:
            final_image = Image.open(camera_file).convert("RGB")
            
    # Preview Gambar Terpilih
    if final_image is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(final_image, caption="Sampel Citra Daun Terpilih Siap Dianalisis", use_container_width=True)
        
    st.markdown("---")
    
    # Tombol Aksi Eksekusi Logika Model
    if st.button("⚙️ Jalankan Analisis Deep Learning", type="primary", use_container_width=True):
        if final_image is None:
            st.warning("⚠️ Peringatan: Mohon masukkan foto daun kopi (lewat unggah berkas ataupun kamera) terlebih dahulu sebelum memulai.")
        else:
            with st.spinner("Sistem sedang melakukan ekstraksi fusi matriks visual dan tabular..."):
                try:
                    # 1. Prapemrosesan Gambar Terpilih
                    pixel_values = image_processor(images=final_image, return_tensors="pt")['pixel_values'].to(device)
                    
                    # 2. Prapemrosesan Tabular
                    raw_tabular = [[input_udara, input_tanah, input_ph]]
                    scaled_tabular = scaler.transform(raw_tabular)
                    tabular_data = torch.tensor(scaled_tabular, dtype=torch.float32).to(device)
                    
                    # 3. Proses Prediksi Model
                    with torch.no_grad():
                        logits = model(pixel_values, tabular_data)
                        probabilities = torch.softmax(logits, dim=1)
                        prediction_idx = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][prediction_idx].item() * 100
                        
                    # 4. Ambil Nama Penyakit Asli dari Label Encoder
                    result_disease = le.inverse_transform([prediction_idx])[0]
                    
                    # 5. TAMPILAN PANEL HASIL UTAMA
                    st.markdown("### 📊 Hasil Analisis Prediksi Sistem")
                    c_hasil, c_skor = st.columns(2)
                    with c_hasil:
                        if str(result_disease).lower().strip() == "jamur lain":
                            display_name = "Embun Jelaga"
                        else:
                            display_name = str(result_disease)
                        st.metric(label="Diagnosa Kesehatan/Penyakit", value=display_name)
                    with c_skor:
                        st.metric(label="Akurasi Keyakinan (Confidence Score)", value=f"{confidence:.2f} %")
                        
                    st.markdown("---")
                    
                    # 6. LOGIKA PENCARIAN KEY DENGAN STRATEGI LOWERCASE & REDIRECT EMBUN JELAGA
                    search_key = str(result_disease).lower().strip()
                    if search_key == "jamur lain":
                        search_key = "embun jelaga"
                        
                    info_pakar = SOLUSI_PENYAKIT.get(search_key, None)
                    
                    if info_pakar is not None:
                        st.subheader(f"💡 Rekomendasi Solusi & Tindakan Pakar: {search_key.title()}")
                        st.caption(f"*Patogen / Nama Ilmiah:* ***{info_pakar['nama_latin']}***")
                        st.markdown(f"**📝 Deskripsi Gejala Klinis:** \n{info_pakar['deskripsi']}")
                        
                        st.markdown("**🛡️ Langkah Pencegahan & Isolasi Penularan Kebun:**")
                        for pencegahan_item in info_pakar['pencegahan']:
                            st.write(f"- {pencegahan_item}")
                            
                        st.markdown("**🛠️ Langkah Tindakan Perbaikan / Rekomendasi Pupuk:**")
                        for perbaikan_item in info_pakar['perbaikan']:
                            st.write(f"- {perbaikan_item}")
                            
                        st.markdown("**💊 Rekomendasi Pengobatan (Bahan Aktif Racun / Fungisida / Insektisida):**")
                        st.info(info_pakar['obat_kimia'])
                    else:
                        st.warning("⚠️ Data rekomendasi penanganan medis untuk kelas penyakit ini belum terdaftar di dalam sistem.")
                        
                except Exception as e:
                    st.error(f"Terjadi kendala komputasi internal: {e}")