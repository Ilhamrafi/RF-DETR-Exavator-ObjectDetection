# frontend/components/data_visualisasi.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

def plot_passing_ritase_time_series(passing_events: List[Dict], ritase_events: List[Dict]):
    """
    Gambar deret waktu (kumulatif) untuk event Passing dan Ritase, lengkap dengan penanda
    garis vertikal pada setiap Ritase.

    Args:
        passing_events: Daftar event bertipe "Passing" (berisi minimal kolom "Detik").
        ritase_events: Daftar event bertipe "Ritase" (berisi minimal kolom "Detik").
    """
    # Ubah list event menjadi DataFrame agar plotting lebih mudah.
    df_passing = pd.DataFrame(passing_events)
    df_ritase = pd.DataFrame(ritase_events)

    # Siapkan figure Plotly: dua trace untuk Passing dan Ritase.
    fig = go.Figure()

    if not df_passing.empty:
        fig.add_trace(go.Scatter(
            x=df_passing["Detik"],
            y=range(1, len(df_passing) + 1),  # kumulatif sederhana: indeks + 1
            mode="lines+markers",
            name="Passing",
            line=dict(color="blue"),
            marker=dict(size=10),
        ))

    if not df_ritase.empty:
        fig.add_trace(go.Scatter(
            x=df_ritase["Detik"],
            y=range(1, len(df_ritase) + 1),  # kumulatif sederhana: indeks + 1
            mode="lines+markers",
            name="Ritase",
            line=dict(color="red"),
            marker=dict(size=12, symbol="star"),
        ))

        # Garis vertikal dan anotasi untuk tiap siklus Ritase.
        for i, row in df_ritase.iterrows():
            fig.add_shape(
                type="line",
                x0=row["Detik"], y0=0,
                x1=row["Detik"], y1=len(df_passing) + 1,
                line=dict(color="gray", dash="dot", width=1),
            )
            fig.add_annotation(
                x=row["Detik"], y=0,
                text=f"Siklus {i+1}",
                showarrow=False, yshift=10,
                font=dict(color="red"),
            )

    # Tata letak umum.
    fig.update_layout(
        title="Passing dan Ritase vs Waktu (dengan Penanda Siklus)",
        xaxis_title="Waktu (detik)",
        yaxis_title="Jumlah Kumulatif",
        legend=dict(y=0.99, x=0.01),
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_passing_per_cycle(df_tracking: pd.DataFrame):
    """
    Tampilkan jumlah Passing per siklus (ambil nilai maksimum per ID siklus).

    Args:
        df_tracking: DataFrame berisi kolom "ID" dan "Passing".
    """
    if df_tracking.empty:
        st.warning("Tidak ada data untuk divisualisasikan")
        return

    # Ambil Passing maksimum per ID (1 baris per siklus).
    cycle_stats = df_tracking.groupby("ID")["Passing"].max().reset_index()

    # Bar chart jumlah Passing per siklus.
    fig = px.bar(
        cycle_stats,
        x="ID", y="Passing",
        title="Jumlah Passing per Siklus",
        labels={"ID": "ID Siklus", "Passing": "Jumlah Passing"},
        color="Passing",
        color_continuous_scale=px.colors.sequential.Blues,
    )
    fig.update_layout(xaxis_type="category")

    st.plotly_chart(fig, use_container_width=True)

def display_excel_summary(excel_path: str):
    """
    Tampilkan ringkasan laporan dari file Excel (info video, data tracking, dan grafik).

    Args:
        excel_path: Path file Excel hasil analisis.
    """
    try:
        # Sheet utama yang diharapkan.
        df_tracking = pd.read_excel(excel_path, sheet_name="Tracking")
        df_info = pd.read_excel(excel_path, sheet_name="Info Video")

        # Sheet opsional (abaikan bila tidak ada).
        try:
            df_detail = pd.read_excel(excel_path, sheet_name="Detail Events")
        except Exception:
            df_detail = None

        # Ringkasan info video (durasi, total frame/passing/ritase).
        st.subheader("Informasi Video")
        info_dict = dict(zip(df_info.iloc[:, 0], df_info.iloc[:, 1]))
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Durasi Video", info_dict.get("Durasi", "N/A"))
            st.metric("Total Frame", info_dict.get("Total Frame", "N/A"))
        with col2:
            st.metric("Total Passing", info_dict.get("Total Passing", "N/A"))
            st.metric("Total Ritase", info_dict.get("Total Ritase", "N/A"))

        # Tabel data mentah tracking.
        st.subheader("Data Tracking")
        st.dataframe(df_tracking, use_container_width=True)

        # Visualisasi agregasi per-ID (maksimum Ritase/Passing per siklus).
        st.subheader("Visualisasi")
        if not df_tracking.empty:
            st.subheader("Passing per Siklus")
            plot_passing_per_cycle(df_tracking)

            # Bar chart Ritase vs Passing per ID.
            id_stats = df_tracking.groupby("ID").agg({
                "Ritase": "max",
                "Passing": "max",
                "Siklus": "max",  # jika tersedia
            }).reset_index()

            fig = px.bar(
                id_stats,
                x="ID",
                y=["Ritase", "Passing"],
                barmode="group",
                title="Ritase dan Passing per ID",
                labels={"value": "Jumlah", "variable": "Tipe", "ID": "ID Siklus"},
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        # Tampilkan error singkat agar mudah ditelusuri di UI.
        st.error(f"Gagal memuat data Excel: {str(e)}")
