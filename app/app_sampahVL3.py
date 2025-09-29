# file: app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import gdown
import pickle

# Link gdrive model (format gdown)
file_id = '1c8l9hD7y4A6kpPmrcZqO_bhPzF5oNEIx'  # ganti dengan ID kamu
gdrive_url = f'https://drive.google.com/uc?id={file_id}'

# Download ke local
output_path = 'best_model_efficientnet.keras'

@st.cache_resource
def load_model():
    gdown.download(gdrive_url, output_path, quiet=False)
    with open(output_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

	
# st.cache_data.clear()
# st.cache_resource.clear()

# MODEL_REPO = "syarifahsgu/rewaste_model_efficientnet"
# MODEL_FILENAME = "best_mobilenetv2_data_split.h5"

# # -------------------------------
# # Load model dari HF
# # -------------------------------
# @st.cache_resource(show_spinner=True)
# def load_model_from_hf():
#     # download model dari HF (public) tanpa token
#     model_path = hf_hub_download(
#         repo_id=MODEL_REPO,
#         filename=MODEL_FILENAME,
#         token=None  # karena public
#     )
#     return load_model(model_path)

# model = load_model_from_hf()

# -------------------------------
# Kelas & Info Pengelolaan
# -------------------------------
class_names = ['battery', 'glass', 'metal', 'organic', 'paper', 'plastic']

info_pengelolaan = {
    'battery': """🔋 Battery (Baterai)

**Kategori**: Sampah elektronik berbahaya (e-waste)  
**Risiko**: Mengandung bahan kimia berbahaya dan bahaya kebakaran  

**Pengelolaan**:
- Jangan dibuang di tempat sampah biasa  
- Bawa ke pusat pengumpulan limbah elektronik resmi  
- Dapat didaur ulang untuk mengambil logam dan bahan kimia  

**Cara daur ulang**:
- Pisahkan berdasarkan jenis (alkaline, lithium, dll)  
- Kirim ke fasilitas daur ulang e-waste  
- Diambil logam dan kimia untuk bahan baku baru
""",
    'glass': """🧪 Glass (Kaca)

**Kategori**: Sampah anorganik yang bisa didaur ulang  
**Risiko**: Pecahan kaca bisa melukai dan sulit terurai di alam  

**Pengelolaan**:
- Pisahkan dari sampah lain  
- Bersihkan sisa makanan/minuman  

**Cara daur ulang**:
- Hancurkan dan lelehkan menjadi bahan baru (botol, keramik)
""",
    'metal': """🥫 Metal (Logam)

**Kategori**: Sampah anorganik bernilai ekonomi tinggi  
**Risiko**: Mencemari tanah dan air jika dibuang sembarangan  

**Pengelolaan**:
- Bersihkan dan pisahkan jenis logam  
- Bawa ke tempat pengumpulan logam  

**Cara daur ulang**:
- Dilebur ulang menjadi bahan mentah  
- Hemat energi dibanding produksi dari bijih
""",
    'organic': """🌿 Organic (Organik)

**Kategori**: Sampah yang mudah terurai alami  
**Risiko**: Menimbulkan bau dan penyakit jika tidak diolah  

**Pengelolaan**:
- Pisahkan dari sampah anorganik  
- Manfaatkan untuk kompos atau biogas  

**Cara daur ulang**:
- Komposkan di rumah atau fasilitas  
- Digunakan sebagai pupuk alami
""",
    'paper': """📄 Paper (Kertas)

**Kategori**: Sampah anorganik mudah didaur ulang  
**Risiko**: Tidak besar, tapi boros jika tidak didaur ulang  

**Pengelolaan**:
- Kumpulkan yang bersih dan kering  
- Hindari kertas berminyak  

**Cara daur ulang**:
- Dicacah dan dicetak ulang jadi kertas baru  
- Bisa juga dijadikan kompos
""",
    'plastic': """♻️ Plastic (Plastik)

**Kategori**: Sampah anorganik yang sulit terurai  
**Risiko**: Polusi besar, terutama di laut  

**Pengelolaan**:
- Pisahkan berdasarkan jenis (PET, HDPE, dll)  
- Cuci dan keringkan  

**Cara daur ulang**:
- Diolah jadi pellet plastik  
- Digunakan untuk barang baru (tas, botol, bahan bangunan)
"""
}


# -------------------------------
# Preprocess gambar
# -------------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# Setup session state
# -------------------------------
if "riwayat" not in st.session_state:
    st.session_state["riwayat"] = []

# -------------------------------
# HEADER LOGO
# -------------------------------
st.markdown(
"""
<div style='display: flex; justify-content: flex-end; gap: 20px; margin-bottom:10px;'>
    <img src='https://4.bp.blogspot.com/-A9bdjWv1xXQ/W68qLVIBoJI/AAAAAAAAGd0/2k4GBWzrJCsR-FDPfKY_oTDh3yhyHwrMgCLcBGAs/s1600/logo-universitas-bina-sarana-informatika-ubsi.png' style='height:50px;'>
    <img src='https://bipemas.bsi.ac.id/assets/images/bipemas_21.png' style='height:50px;'>
    <img src='https://ejournal.bsi.ac.id/ejurnal/public/site/images/piyan/Logo-LPPM-UBSI.png' style='height:50px;'>
</div>
""",
unsafe_allow_html=True
)

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.markdown("<h4 style='color: darkblue;'>Tentang Sistem</h5>", unsafe_allow_html=True)
    st.sidebar.info(
        """
        Sistem ini membantu pengguna mengklasifikasikan jenis sampah
        otomatis menggunakan Deep Learning.
        """
    )
    st.markdown("<h4 style='color: darkblue;'>Kategori Sampah</h5>", unsafe_allow_html=True)
    for kls in class_names:
        st.sidebar.markdown(f"• {kls.capitalize()}")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Beranda", "Klasifikasi", "Tentang", "Kontak"])

# ===== Tab 1 =====
with tab1:
    st.image("banner.png", use_container_width=True)
    st.markdown("---")
    st.markdown("<h6 style='color: darkblue;'>Selamat Datang di Sistem Klasifikasi Sampah ♻️</h6>", unsafe_allow_html=True)
    st.write("Upload gambar sampah, sistem akan memprediksi kategori dan info pengelolaan.")

# ===== Tab 2 =====
with tab2:
    st.markdown("<h1 style='text-align: center; color: #4cd137;'>♻️ Klasifikasi Sampah</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: darkblue'>Upload gambar sampah untuk mengetahui jenis dan cara pengelolaannya</h4>", unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Upload gambar sampah", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown("<h6 style='color: darkblue;'>📷 Gambar yang Diunggah</h6>", unsafe_allow_html=True)
            st.image(img, width=250)
        with col2:
            processed_img = preprocess_image(img)
            prediction = model.predict(processed_img)
            pred_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
            st.markdown(f"<h5 style='color: darkblue;'>🧠 Hasil Prediksi: {pred_class.capitalize()} ({confidence:.2f})</h5>", unsafe_allow_html=True)
            st.success(info_pengelolaan[pred_class])
            
            # Simpan ke riwayat
            st.session_state["riwayat"].append({
                "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "File": uploaded_file.name,
                "Prediksi": pred_class.capitalize(),
                "Akurasi": round(float(confidence), 2)
            })
        
        # Riwayat prediksi
        st.markdown("---")
        st.markdown("<h5 style='color: darkblue;'>📜 Riwayat Prediksi</h5>", unsafe_allow_html=True)
        df = pd.DataFrame(st.session_state["riwayat"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Silakan upload gambar terlebih dahulu untuk melihat hasil klasifikasi.")

# ===== Tab 3 =====
with tab3:
    st.markdown("<h4 style='color: #4cd137;'>Tentang Sistem</h5>", unsafe_allow_html=True)
    st.write(
        """
        Sistem ini membantu mengklasifikasikan sampah secara otomatis menggunakan Deep Learning.
        Upload gambar sampah, sistem akan memprediksi kategori dan memberikan info pengelolaan.
        """
    )

# ===== Tab 4 =====
with tab4:
    st.markdown("<h4 style='color: #4cd137;'>Kontak Kami:</h5>", unsafe_allow_html=True)
    st.write("1. Sarifah Agustiani\n2. Haryani\n3. Agus Junaisi")
    st.write("Program ini merupakan luaran penelitian dosen yayasan, mendukung inovasi teknologi dan pengelolaan sampah berkelanjutan.")

# ===== Footer =====
st.markdown("---")
st.markdown(
"""
<div style='text-align: center; color: #4cd137; font-size: 14px; margin-top:20px;'>
<b>Sarifah Agustiani, Haryani, Agus Junaidi</b> <br>
Penelitian Dosen Yayasan | Universitas Bina Sarana Informatika | 2025
</div>
""",
unsafe_allow_html=True
)





# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image
# from tensorflow.keras.applications.efficientnet import preprocess_input
# import pandas as pd
# from datetime import datetime

# hf-auth-login

# # === Setup session state untuk riwayat ===
# if "riwayat" not in st.session_state:
#     st.session_state["riwayat"] = []
    
# # Available backend options are: "jax", "torch", "tensorflow".
# import os
# os.environ["KERAS_BACKEND"] = "jax"
	
# import keras

# model = keras.saving.load_model("hf://syarifahsgu/rewaste_model_efficientnet")

# class_names = ['battery', 'glass', 'metal', 'organic', 'paper', 'plastic']

# info_pengelolaan = {
#     'battery': """🔋 Battery (Baterai)

# **Kategori**: Sampah elektronik berbahaya (e-waste)  
# **Risiko**: Mengandung bahan kimia berbahaya dan bahaya kebakaran  

# **Pengelolaan**:
# - Jangan dibuang di tempat sampah biasa  
# - Bawa ke pusat pengumpulan limbah elektronik resmi  
# - Dapat didaur ulang untuk mengambil logam dan bahan kimia  

# **Cara daur ulang**:
# - Pisahkan berdasarkan jenis (alkaline, lithium, dll)  
# - Kirim ke fasilitas daur ulang e-waste  
# - Diambil logam dan kimia untuk bahan baku baru
# """,
#     'glass': """🧪 Glass (Kaca)

# **Kategori**: Sampah anorganik yang bisa didaur ulang  
# **Risiko**: Pecahan kaca bisa melukai dan sulit terurai di alam  

# **Pengelolaan**:
# - Pisahkan dari sampah lain  
# - Bersihkan sisa makanan/minuman  

# **Cara daur ulang**:
# - Hancurkan dan lelehkan menjadi bahan baru (botol, keramik)
# """,
#     'metal': """🥫 Metal (Logam)

# **Kategori**: Sampah anorganik bernilai ekonomi tinggi  
# **Risiko**: Mencemari tanah dan air jika dibuang sembarangan  

# **Pengelolaan**:
# - Bersihkan dan pisahkan jenis logam  
# - Bawa ke tempat pengumpulan logam  

# **Cara daur ulang**:
# - Dilebur ulang menjadi bahan mentah  
# - Hemat energi dibanding produksi dari bijih
# """,
#     'organic': """🌿 Organic (Organik)

# **Kategori**: Sampah yang mudah terurai alami  
# **Risiko**: Menimbulkan bau dan penyakit jika tidak diolah  

# **Pengelolaan**:
# - Pisahkan dari sampah anorganik  
# - Manfaatkan untuk kompos atau biogas  

# **Cara daur ulang**:
# - Komposkan di rumah atau fasilitas  
# - Digunakan sebagai pupuk alami
# """,
#     'paper': """📄 Paper (Kertas)

# **Kategori**: Sampah anorganik mudah didaur ulang  
# **Risiko**: Tidak besar, tapi boros jika tidak didaur ulang  

# **Pengelolaan**:
# - Kumpulkan yang bersih dan kering  
# - Hindari kertas berminyak  

# **Cara daur ulang**:
# - Dicacah dan dicetak ulang jadi kertas baru  
# - Bisa juga dijadikan kompos
# """,
#     'plastic': """♻️ Plastic (Plastik)

# **Kategori**: Sampah anorganik yang sulit terurai  
# **Risiko**: Polusi besar, terutama di laut  

# **Pengelolaan**:
# - Pisahkan berdasarkan jenis (PET, HDPE, dll)  
# - Cuci dan keringkan  

# **Cara daur ulang**:
# - Diolah jadi pellet plastik  
# - Digunakan untuk barang baru (tas, botol, bahan bangunan)
# """
# }

# def preprocess_image(img):
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # ====== HEADER LOGO ======
# st.markdown(
#         """
#         <div style='display: flex; justify-content: flex-end; gap: 20px; margin-bottom:10px;'>
#             <img src='https://4.bp.blogspot.com/-A9bdjWv1xXQ/W68qLVIBoJI/AAAAAAAAGd0/2k4GBWzrJCsR-FDPfKY_oTDh3yhyHwrMgCLcBGAs/s1600/logo-universitas-bina-sarana-informatika-ubsi.png' style='height:50px;'>
#             <img src='https://bipemas.bsi.ac.id/assets/images/bipemas_21.png' style='height:50px;'>
#             <img src='https://ejournal.bsi.ac.id/ejurnal/public/site/images/piyan/Logo-LPPM-UBSI.png' style='height:50px;'>
#         </div>
#         """,
#         unsafe_allow_html=True)


# with st.sidebar:
 
#     st.markdown("<h4 style='color: darkblue;'>Tentang Sistem</h5>", unsafe_allow_html=True)
#     st.sidebar.info(
#        """
#         Sistem ini dirancang untuk membantu pengguna dalam **mengklasifikasikan jenis sampah** 
#         secara otomatis menggunakan teknologi **Deep Learning**.  
#         """
#    )
#     st.markdown("<h4 style='color: darkblue;'>Kategori Sampah</h5>", unsafe_allow_html=True)
#     for kls in class_names:
#         st.sidebar.markdown(
#         f"""
#         <div style='
#             font-size:14px;
#             padding:3px 6px;
#             border-radius:4px;
#             margin-bottom:3px;
#         '>
#             •  {kls.capitalize()}
#         </div>
#         """, unsafe_allow_html=True)
  

# # ====== TAB MENU ======
# tab1, tab2, tab3, tab4 = st.tabs(["Beranda", "Klasifikasi", "Tentang", "Kontak"])

# with tab1:
#     st.image("banner.png", use_container_width=True) 
#     st.markdown("---")
#     st.markdown("<h6 style='color: darkblue;'>Selamat Datang di Sistem Klasifikasi Sampah ♻️</h6>",unsafe_allow_html=True)
#     st.write("Website ini dibuat untuk mengklasifikasikan jenis sampah dan memberikan informasi pengelolaannya.")      


# with tab2:
#     st.markdown("<h1 style='text-align: center; color: #4cd137;'>♻️ Klasifikasi Sampah</h1>", unsafe_allow_html=True)
#     st.markdown("<h4 style='text-align: center; color: darkblue'>Upload gambar sampah untuk mengetahui jenis dan cara pengelolaannya</h4>", unsafe_allow_html=True)
#     st.markdown("---")

# # === Upload file ===
#     uploaded_file = st.file_uploader("📤 Upload gambar sampah", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         img = Image.open(uploaded_file)

#         # Layout 2 kolom: kiri gambar, kanan hasil prediksi
#         col1, col2 = st.columns([1,2])

#         with col1:
#             st.markdown("<h6 style='color: darkblue;'>📷 Gambar yang Diunggah</h6>",unsafe_allow_html=True)
#             st.image(img, width=250)

#         with col2:
#             processed_img = preprocess_image(img)
#             prediction = model.predict(processed_img)
#             pred_class = class_names[np.argmax(prediction)]
#             confidence = np.max(prediction)

#             st.markdown(f"<h5 style='color: darkblue;'>🧠 Hasil Prediksi: {pred_class.capitalize()} ({confidence:.2f})</h5>",unsafe_allow_html=True)
#             st.success(info_pengelolaan[pred_class])

#             # Simpan ke riwayat
#             st.session_state["riwayat"].append({
#                 "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "File": uploaded_file.name,
#                 "Prediksi": pred_class.capitalize(),
#                 "Akurasi": round(float(confidence), 2)
#             })

#         # === Riwayat Prediksi ===
#         st.markdown("---")
#         st.markdown(f"<h5 style='color: darkblue;'> 📜 Riwayat Prediksi",unsafe_allow_html=True)
#         df = pd.DataFrame(st.session_state["riwayat"])
#         st.dataframe(df, use_container_width=True)

#     else:
#         st.info("Silakan upload gambar terlebih dahulu untuk melihat hasil klasifikasi.")

# with tab3:
#     st.markdown("<h4 style='color: #4cd137;'>Tentang Sistem</h5>", unsafe_allow_html=True)
#     st.markdown(
#        """
#         Sistem ini dirancang untuk membantu pengguna dalam **mengklasifikasikan jenis sampah** 
#         secara otomatis menggunakan teknologi **Deep Learning**.  
#         Dengan mengunggah gambar sampah, sistem akan memberikan hasil prediksi kategori beserta 
#         informasi singkat mengenai cara **pengelolaan dan daur ulang** yang tepat.  

#         Tujuan utama dari sistem ini adalah untuk:
#         - Meningkatkan kesadaran masyarakat terhadap pentingnya pemilahan sampah.
#         - Mendukung pengelolaan limbah yang berkelanjutan.
#         - Memanfaatkan teknologi kecerdasan buatan guna menciptakan solusi ramah lingkungan.
#         """, unsafe_allow_html=True)
#     st.markdown("<h4 style='color: #f6e58d;'>Kategori Sampah</h5>", unsafe_allow_html=True)
#     for kls in class_names:
#         st.markdown(f"- {kls.capitalize()}")

# with tab4:
#     st.markdown("<h4 style='color: #4cd137;'>Kontak Kami:</h5>", unsafe_allow_html=True)
#     st.write("1. Sarifah Agustiani\n2. Haryani\n3. Agus Junaisi")
#     st.write("Aplikasi ini merupakan salah satu luaran dari program pendanaan yang difasilitasi oleh yayasan, yang bertujuan untuk mendorong pengembangan inovasi dan implementasi teknologi di masyarakat.")

# # Footer
# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center; color: #4cd137; font-size: 14px; margin-top:20px;'>
#         <b>Sarifah Agustiani, Haryani, Agus Junaidi</b> <br>
#         Penelitian Dosen Yayasan | Universitas Bina Sarana Informatika | 2025
#     </div>
#     """,
#     unsafe_allow_html=True
# )
