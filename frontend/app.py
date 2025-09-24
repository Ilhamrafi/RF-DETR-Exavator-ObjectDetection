# frontend/app.py

"""
Aplikasi Streamlit untuk Sistem Deteksi Excavator.
"""

import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os

# Tambahkan path root project ke sys.path agar impor modul lokal lebih mudah.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Impor fungsi view untuk mengontrol tampilan per-halaman.
from frontend.view.detect_view import show_detect_page
from frontend.view.results_view import show_results_page

# Konfigurasi halaman Streamlit.
st.set_page_config(
    page_title="Excavator Detection System",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sembunyikan hamburger menu, footer, dan navigasi sidebar bawaan Streamlit.
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# CSS untuk tema dan komponen UI.
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #ffffff;
        text-align: center;
    }
    .main-subheader {
        font-size: 1.2rem;
        color: #aaaaaa;
        margin-bottom: 2rem;
        text-align: center;
    }
    /* Tampilan tab aktif */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #333333;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        min-width: 150px;
        text-align: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF5151;
        color: white !important;
        font-weight: bold;
    }
    /* Pastikan label tab terbaca */
    .stTabs [data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p {
        color: white !important;
        font-weight: bold;
        margin: 0;
    }
    /* Gaya item menu pada sidebar */
    .nav-link {
        margin: 0.2rem 0;
        border-radius: 5px;
    }
    .nav-link-selected {
        background-color: #FF5151 !important;
        color: white !important;
    }
    /* Gaya tombol utama */
    .stButton button {
        background-color: #FF5151;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #FF7070;
    }
    /* Kartu info */
    .info-card {
        background-color: #1D2530;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header utama.
st.markdown('<div class="main-header">Sistem Deteksi Excavator</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subheader">Deteksi aktivitas excavator dan truck</div>', unsafe_allow_html=True)

# Sidebar: kontrol navigasi dan sinkronisasi state halaman.
with st.sidebar:
    # Buat state "page" jika belum ada (default: Deteksi Video).
    if 'page' not in st.session_state:
        st.session_state.page = "Deteksi Video"
        
    # Simpan nilai sebelumnya untuk mendeteksi perubahan pilihan.
    previous_page = st.session_state.page

    # Menu navigasi menggunakan option_menu.
    selected = option_menu(
        "Menu",
        ["Deteksi Video", "Hasil Deteksi"],
        icons=["camera-video-fill", "graph-up"],
        menu_icon="list",
        default_index=0 if previous_page == "Deteksi Video" else 1,
        styles={
            "container": {"padding": "0!important", "background-color": "#262730"},
            "icon": {"color": "#FF5151", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "padding": "10px",
                "--hover-color": "#363B46",
            },
            "nav-link-selected": {"background-color": "#FF5151"},
        },
    )
    
    # Perbarui state dan rerun bila ada perubahan halaman.
    if selected != previous_page:
        st.session_state.page = selected
        st.rerun()
    
    st.sidebar.markdown("---")
    # Informasi aplikasi.
    # st.sidebar.info("Aplikasi ini mendeteksi aktivitas excavator dan menghitung passing/ritase dari video input.")

# Routing halaman berdasarkan state.
if st.session_state.page == "Deteksi Video":
    show_detect_page()
else:
    show_results_page()