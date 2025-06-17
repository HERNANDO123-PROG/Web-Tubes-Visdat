import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter
from bokeh.transform import cumsum, factor_cmap, linear_cmap
from bokeh.palettes import Category10, Category20c, Viridis256
from math import pi

st.set_page_config(page_title="Analisis Konsumsi Energi Global", layout="wide")

st.title("Analisis Konsumsi Energi Global dengan Clustering AI & Visualisasi Interaktif")

# Load full data initially to populate filters
@st.cache_data
def load_full_data():
    df_full = pd.read_csv("global_energy_consumption.csv")
    df_agglo_full = pd.read_csv("hasil_agglo_clustering.csv")
    return df_full, df_agglo_full

df_full, df_agglo_full = load_full_data()

# Sidebar
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "Eksplorasi Data",
    "Visualisasi Interaktif",
    "Analisis Clustering (AI)"
])

# --- Filter Global ---
st.sidebar.subheader("Filter Data")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Data Energi Global (CSV)", type=["csv"])

@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file):
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.sidebar.success("Data berhasil diunggah dan dimuat!")
        return df_uploaded
    except Exception as e:
        st.sidebar.error(f"Error memuat file: {e}")
        return None

if uploaded_file is not None:
    df_full = load_data_from_upload(uploaded_file)
    if df_full is None:
        # Fallback to original data if upload fails
        df_full = pd.read_csv("global_energy_consumption.csv")
        df_agglo_full = pd.read_csv("hasil_agglo_clustering.csv") # Revert df_agglo_full as well
        st.sidebar.warning("Menggunakan data default karena file yang diunggah bermasalah.")
else:
    df_full = pd.read_csv("global_energy_consumption.csv")
    df_agglo_full = pd.read_csv("hasil_agglo_clustering.csv")

# Perbarui all_years dan all_countries berdasarkan df_full yang mungkin baru
all_years = sorted(df_full['Year'].unique())
selected_years = st.sidebar.slider(
    "Pilih Rentang Tahun",
    min_value=int(min(all_years)),
    max_value=int(max(all_years)),
    value=(int(min(all_years)), int(max(all_years)))
)

all_countries = sorted(df_full['Country'].unique())
selected_countries = st.sidebar.multiselect(
    "Pilih Negara (Visualisasi Interaktif)",
    options=all_countries,
    default=all_countries[:10] # Default 10 negara pertama
)

# Apply filters based on sidebar selection
df = df_full[(df_full['Year'] >= selected_years[0]) & (df_full['Year'] <= selected_years[1])]
df = df[df['Country'].isin(selected_countries)]

# Note: df_agglo tetap menggunakan data asli hasil_agglo_clustering.csv
# Jika ingin clustering ulang dengan data yang diupload, perlu kode AI clustering di sini.
# Untuk saat ini, df_agglo tidak difilter berdasarkan negara dari uploaded_file.
df_agglo = df_agglo_full[(df_agglo_full['Year'] >= selected_years[0]) & (df_agglo_full['Year'] <= selected_years[1])]
df_agglo = df_agglo[df_agglo['Country'].isin(selected_countries)] # Filter df_agglo based on selected countries

# Pastikan df_agglo memiliki kolom Cluster sebagai string untuk visualisasi
df_agglo["Cluster"] = df_agglo["Cluster"].astype(str)

if menu == "Eksplorasi Data":
    st.header("Eksplorasi Data Energi Global")
    st.markdown("""
    **Sumber Dataset:** [Global Energy Consumption 2000-2024 (Kaggle)](https://www.kaggle.com/datasets/atharvasoundankar/global-energy-consumption-2000-2024)
    """)
    st.markdown("""
    Pada bagian ini, Anda dapat mengeksplorasi data konsumsi energi global secara deskriptif. Eksplorasi ini bertujuan untuk memahami karakteristik dasar data, distribusi, korelasi antar fitur, serta mendeteksi outlier yang dapat mempengaruhi analisis lanjutan. Statistik deskriptif, distribusi per tahun dan negara, serta visualisasi korelasi dan outlier membantu mengidentifikasi pola umum dan potensi masalah pada data.
    """)
    st.subheader("Statistik Deskriptif")
    st.info("""
    Statistik deskriptif memberikan gambaran umum mengenai sebaran nilai, rata-rata, minimum, maksimum, dan standar deviasi dari setiap fitur numerik dalam data energi global. Hal ini penting untuk memahami skala dan variasi data sebelum melakukan analisis lebih lanjut.
    """)
    st.dataframe(df.describe())

    st.subheader("Download Data Hasil Filter")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Unduh Data Energi Global (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="global_energy_consumption_filtered.csv",
            mime="text/csv",
            help="Unduh data energi global yang saat ini difilter."
        )
    with col2:
        st.download_button(
            label="Unduh Data Klastering (CSV)",
            data=df_agglo.to_csv(index=False).encode('utf-8'),
            file_name="hasil_agglo_clustering_filtered.csv",
            mime="text/csv",
            help="Unduh data hasil clustering (AI) yang saat ini difilter."
        )

    st.subheader("Distribusi Jumlah Data Per Tahun")
    st.info("""
    Visualisasi ini menunjukkan jumlah entri data untuk setiap tahun. Dengan melihat distribusi ini, kita dapat mengetahui apakah data terdistribusi merata sepanjang waktu atau terdapat tahun-tahun tertentu dengan data lebih banyak/lebih sedikit.
    """)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(x='Year', data=df, palette="Set3", ax=ax)
    ax.set_title("Distribusi Jumlah Data Per Tahun")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.subheader("10 Negara dengan Jumlah Data Terbanyak")
    st.info("""
    Grafik batang ini menampilkan 10 negara dengan jumlah data terbanyak dalam dataset. Hal ini membantu mengidentifikasi negara-negara yang paling sering tercatat dan dapat menjadi fokus analisis lebih lanjut.
    """)
    fig, ax = plt.subplots(figsize=(10, 4))
    df["Country"].value_counts().head(10).plot(kind='bar', color='teal', ax=ax)
    ax.set_title("10 Negara dengan Jumlah Data Terbanyak")
    ax.set_ylabel("Jumlah Data")
    st.pyplot(fig)

    st.subheader("Korelasi Antar Variabel Numerik")
    st.info("""
    Heatmap korelasi ini memperlihatkan hubungan linear antar fitur numerik utama dalam data energi global. Korelasi positif/negatif yang kuat dapat mengindikasikan adanya keterkaitan antar fitur, yang penting untuk analisis lanjutan dan pemodelan.
    """)
    fig, ax = plt.subplots(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include='number').columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Korelasi Antar Variabel Numerik")
    st.pyplot(fig)

    st.subheader("Distribusi Fitur Numerik")
    st.info("""
    Histogram ini menampilkan distribusi nilai dari setiap fitur numerik. Dengan melihat histogram, kita dapat mengetahui apakah data berdistribusi normal, skewed, atau memiliki outlier.
    """)
    df_numeric = df.select_dtypes(include='number')
    fig, ax = plt.subplots(figsize=(15, 10))
    df_numeric.hist(bins=30, color='salmon', edgecolor='black', ax=ax)
    plt.suptitle("Distribusi Fitur Numerik", fontsize=16)
    st.pyplot(fig)

    st.subheader("Deteksi Outlier Pada Fitur Numerik")
    st.info("""
    Boxplot ini digunakan untuk mendeteksi outlier pada fitur numerik. Outlier dapat mempengaruhi hasil analisis dan pemodelan, sehingga penting untuk diidentifikasi sejak awal.
    """)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=df_numeric, orient="h", palette="pastel", ax=ax)
    ax.set_title("Deteksi Outlier Pada Fitur Numerik")
    st.pyplot(fig)

elif menu == "Visualisasi Interaktif":
    st.header("Visualisasi Interaktif Data Energi Global")
    # 1. Rata-rata konsumsi energi per tahun (Bokeh)
    st.subheader("Rata-Rata Konsumsi Energi Total Per Tahun")
    energy_avg = df.groupby("Year")["Total Energy Consumption (TWh)"].mean().reset_index()
    source = ColumnDataSource(energy_avg)
    p = figure(title="Rata-Rata Konsumsi Energi Total Per Tahun", x_axis_label='Tahun', y_axis_label='Energi (TWh)', width=800, height=400, tools="pan,wheel_zoom,box_zoom,reset,hover,save")
    p.line(x='Year', y='Total Energy Consumption (TWh)', source=source, line_width=2, color="navy")
    p.circle(x='Year', y='Total Energy Consumption (TWh)', source=source, size=6, color="navy", alpha=0.6)
    p.add_tools(HoverTool(tooltips=[("Tahun", "@Year"), ("Energi", "@{Total Energy Consumption (TWh)}{0.0}")]))
    p.title.align = 'center'
    st.bokeh_chart(p, use_container_width=True)

    # 2. Top 10 negara konsumsi energi (Bokeh)
    st.subheader("10 Negara dengan Rata-Rata Konsumsi Energi Tertinggi")
    country_avg = df.groupby("Country")["Total Energy Consumption (TWh)"].mean().reset_index()
    top10 = country_avg.sort_values(by="Total Energy Consumption (TWh)", ascending=False).head(10)
    top10["color"] = Category10[10]
    source2 = ColumnDataSource(top10)
    p2 = figure(x_range=top10["Country"], title="10 Negara dengan Rata-Rata Konsumsi Energi Tertinggi", x_axis_label='Negara', y_axis_label='Rata-rata Konsumsi Energi (TWh)', width=800, height=400, tools="pan,box_zoom,reset,hover,save")
    p2.vbar(x='Country', top='Total Energy Consumption (TWh)', source=source2, width=0.6, fill_color='color')
    p2.add_tools(HoverTool(tooltips=[("Negara", "@Country"), ("Energi", "@{Total Energy Consumption (TWh)}{0.0}")]))
    p2.xaxis.major_label_orientation = 1.0
    p2.title.align = 'center'
    p2.title.text_font_size = '14pt'
    st.bokeh_chart(p2, use_container_width=True)

    # 3. Rata-rata emisi karbon per tahun (Bokeh)
    st.subheader("Rata-Rata Emisi Karbon Global Per Tahun")
    emission_avg = df.groupby("Year")["Carbon Emissions (Million Tons)"].mean().reset_index()
    source3 = ColumnDataSource(emission_avg)
    p3 = figure(title="Rata-rata Emisi Karbon Global Per Tahun", x_axis_label='Tahun', y_axis_label='Emisi Karbon (Juta Ton)', width=800, height=400, tools="pan,box_zoom,reset,hover,save")
    p3.line(x='Year', y='Carbon Emissions (Million Tons)', source=source3, line_width=2, color="firebrick")
    p3.circle(x='Year', y='Carbon Emissions (Million Tons)', source=source3, size=6, color="firebrick", alpha=0.6)
    p3.add_tools(HoverTool(tooltips=[("Tahun", "@Year"), ("Emisi", "@{Carbon Emissions (Million Tons)}{0.0}")]))
    p3.title.align = 'center'
    st.bokeh_chart(p3, use_container_width=True)

    st.subheader("Animasi Tren Emisi Karbon Global Per Tahun")
    st.info("Gunakan slider di bawah untuk melihat tren emisi karbon global dari waktu ke waktu.")
    
    # Slider untuk animasi tahun
    animation_year = st.slider(
        "Pilih Tahun untuk Animasi Emisi Karbon",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=int(df['Year'].min()),
        step=1
    )

    # Filter data hingga tahun yang dipilih untuk animasi
    animated_emission_avg = emission_avg[emission_avg['Year'] <= animation_year]
    source_animated = ColumnDataSource(animated_emission_avg)

    p3_animated = figure(title=f"Tren Emisi Karbon Global Hingga Tahun {animation_year}", x_axis_label='Tahun', y_axis_label='Emisi Karbon (Juta Ton)', width=800, height=400, tools="pan,box_zoom,reset,hover,save")
    p3_animated.line(x='Year', y='Carbon Emissions (Million Tons)', source=source_animated, line_width=2, color="firebrick")
    p3_animated.circle(x='Year', y='Carbon Emissions (Million Tons)', source=source_animated, size=6, color="firebrick", alpha=0.6)
    p3_animated.add_tools(HoverTool(tooltips=[("Tahun", "@Year"), ("Emisi", "@{Carbon Emissions (Million Tons)}{0.0}")]))
    p3_animated.title.align = 'center'
    st.bokeh_chart(p3_animated, use_container_width=True)

    # 4. Top 10 negara energi terbarukan (horizontal bar, Bokeh)
    st.subheader("Top 10 Negara dengan Rata-Rata Proporsi Energi Terbarukan Tertinggi")
    renew_avg = df.groupby("Country")["Renewable Energy Share (%)"].mean().reset_index()
    top10_renew = renew_avg.sort_values(by="Renewable Energy Share (%)", ascending=False).head(10)
    top10_renew["Color"] = Category10[10]
    top10_renew = top10_renew.sort_values("Renewable Energy Share (%)")
    source4 = ColumnDataSource(top10_renew)
    p4 = figure(y_range=top10_renew["Country"], width=800, height=400, title="Top 10 Negara dengan Rata-Rata Proporsi Energi Terbarukan Tertinggi", x_axis_label='Proporsi Energi Terbarukan (%)', tools="pan,box_zoom,reset,hover,save")
    p4.hbar(y='Country', right='Renewable Energy Share (%)', height=0.6, source=source4, fill_color='Color')
    p4.add_tools(HoverTool(tooltips=[("Negara", "@Country"), ("Energi Terbarukan", "@{Renewable Energy Share (%)}{0.0}%")]))
    p4.title.align = 'center'
    st.bokeh_chart(p4, use_container_width=True)

    # 5. Area chart emisi karbon per tahun (Bokeh)
    st.subheader("Tren Rata-Rata Emisi Karbon Global Per Tahun")
    carbon_avg = df.groupby("Year")["Carbon Emissions (Million Tons)"].mean().reset_index()
    source5 = ColumnDataSource(carbon_avg)
    p5 = figure(title="Tren Rata-Rata Emisi Karbon Global Per Tahun", x_axis_label="Tahun", y_axis_label="Emisi Karbon (Juta Ton)", width=800, height=400, tools="pan,box_zoom,reset,hover,save")
    p5.varea(x='Year', y1=0, y2='Carbon Emissions (Million Tons)', source=source5, fill_color="firebrick", fill_alpha=0.5)
    p5.line(x='Year', y='Carbon Emissions (Million Tons)', source=source5, line_width=2, color="firebrick")
    p5.circle(x='Year', y='Carbon Emissions (Million Tons)', source=source5, size=6, color="firebrick")
    p5.add_tools(HoverTool(tooltips=[("Tahun", "@Year"), ("Emisi", "@{Carbon Emissions (Million Tons)}{0.0}")]))
    p5.title.align = 'center'
    st.bokeh_chart(p5, use_container_width=True)

    # 6. Donut chart komposisi energi global (Bokeh)
    st.subheader("Komposisi Rata-Rata Sumber Energi Global")
    renew = df["Renewable Energy Share (%)"].mean()
    fossil = df["Fossil Fuel Dependency (%)"].mean()
    other = 100 - (renew + fossil)
    data = pd.Series({'Energi Terbarukan': renew, 'Bahan Bakar Fosil': fossil, 'Lainnya': other}).reset_index(name='value').rename(columns={'index': 'sumber'})
    data['angle'] = data['value'] / data['value'].sum() * 2 * pi
    data['color'] = Category20c[len(data)]
    source6 = ColumnDataSource(data)
    p6 = figure(height=400, width=400, title="Komposisi Rata-Rata Sumber Energi Global", toolbar_location=None, tools="hover", tooltips="@sumber: @value{0.2f}%", x_range=(-0.5, 1.0))
    p6.wedge(x=0, y=1, radius=0.4, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'), line_color="white", fill_color='color', legend_field='sumber', source=source6)
    p6.annular_wedge(x=0, y=1, inner_radius=0.2, outer_radius=0.4, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'), fill_color='color', line_color="white", source=source6)
    p6.axis.visible = False
    p6.grid.visible = False
    p6.title.align = 'center'
    st.bokeh_chart(p6, use_container_width=False)

    # 7. Heatmap korelasi fitur (Bokeh)
    st.subheader("Heatmap Korelasi Antar Fitur Energi")
    features = [
        "Total Energy Consumption (TWh)",
        "Per Capita Energy Use (kWh)",
        "Renewable Energy Share (%)",
        "Fossil Fuel Dependency (%)",
        "Carbon Emissions (Million Tons)",
        "Energy Price Index (USD/kWh)"
    ]
    corr_matrix = df[features].corr().round(2)
    corr_df = corr_matrix.stack().reset_index()
    corr_df.columns = ["Feature_X", "Feature_Y", "Correlation"]
    source7 = ColumnDataSource(corr_df)
    mapper = LinearColorMapper(palette=Viridis256, low=-1, high=1)
    p7 = figure(title="Heatmap Korelasi Antar Fitur Energi", x_range=features, y_range=list(reversed(features)), x_axis_location="above", width=800, height=600, tools="hover,save", tooltips=[('X', '@Feature_X'), ('Y', '@Feature_Y'), ('Korelasi', '@Correlation')])
    p7.rect(x="Feature_X", y="Feature_Y", width=1, height=1, source=source7, fill_color=linear_cmap('Correlation', Viridis256, -1, 1), line_color=None)
    color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(desired_num_ticks=10), formatter=PrintfTickFormatter(format="%.2f"), label_standoff=12, border_line_color=None, location=(0, 0))
    p7.add_layout(color_bar, 'right')
    p7.xaxis.major_label_orientation = np.pi / 4
    p7.title.align = 'center'
    st.bokeh_chart(p7, use_container_width=True)

    # 8. Scatter plot emisi karbon vs energi terbarukan (top 10 negara, Bokeh)
    st.subheader("Scatter Plot: Emisi Karbon vs Energi Terbarukan (Top 10 Negara)")
    top10_emission = df.groupby("Country")["Carbon Emissions (Million Tons)"].mean().reset_index().sort_values(by="Carbon Emissions (Million Tons)", ascending=False).head(10)
    df_top10 = df[df["Country"].isin(top10_emission["Country"])]
    avg_top10 = df_top10.groupby("Country")[["Carbon Emissions (Million Tons)", "Renewable Energy Share (%)"]].mean().reset_index()
    source8 = ColumnDataSource(avg_top10)
    p8 = figure(title="Scatter Plot: Emisi Karbon vs Energi Terbarukan (Top 10 Negara)", x_axis_label="Proporsi Energi Terbarukan (%)", y_axis_label="Emisi Karbon (Juta Ton)", width=850, height=500, tools="pan,box_zoom,reset,hover,save")
    p8.circle(x='Renewable Energy Share (%)', y='Carbon Emissions (Million Tons)', size=10, source=source8, fill_color=factor_cmap('Country', palette=Category10[10], factors=avg_top10["Country"].tolist()), line_color="black", fill_alpha=0.7, legend_field="Country")
    p8.add_tools(HoverTool(tooltips=[("Negara", "@Country"), ("Energi Terbarukan", "@{Renewable Energy Share (%)}{0.0}%"), ("Emisi Karbon", "@{Carbon Emissions (Million Tons)}{0.0} Juta Ton")]))
    p8.legend.location = "top_right"
    p8.legend.title = "Negara"
    p8.legend.click_policy = "hide"
    p8.title.align = 'center'
    st.bokeh_chart(p8, use_container_width=True)

elif menu == "Analisis Clustering (AI)":
    st.header("Analisis Clustering (AI)")
    st.markdown("""
    Web ini menampilkan analisis konsumsi energi global berbasis AI (Agglomerative Clustering). Negara-negara dikelompokkan berdasarkan pola konsumsi energi, proporsi energi terbarukan, ketergantungan bahan bakar fosil, dan emisi karbon. Setiap visualisasi membantu memahami perbedaan karakteristik energi antar klaster, mendukung pengambilan kebijakan energi yang lebih tepat.
    """)
    # 1. Rata-rata konsumsi energi per tahun berdasarkan klaster
    st.subheader("Rata-Rata Konsumsi Energi Tahunan Berdasarkan Klaster (AI)")
    st.info("""
    Visualisasi ini menunjukkan tren konsumsi energi tahunan rata-rata dari tiap klaster hasil Agglomerative Clustering. Garis berwarna mewakili masing-masing klaster, memperlihatkan perbedaan pola konsumsi energi antar kelompok negara. Beberapa klaster mengalami kenaikan signifikan, sementara lainnya cenderung stabil. Klasterisasi membantu mengidentifikasi pola ini untuk mendukung kebijakan energi yang lebih terarah.
    """)
    df_agglo["Cluster"] = df_agglo["Cluster"].astype(str)
    energy_by_year_cluster = df_agglo.groupby(["Year", "Cluster"])["Total Energy Consumption (TWh)"].mean().reset_index()
    energy_by_year_cluster["Year"] = energy_by_year_cluster["Year"].astype(int)
    clusters = sorted(df_agglo["Cluster"].unique())
    colors = Category10[len(clusters)]
    p9 = figure(title="Rata-rata Konsumsi Energi Tahunan Berdasarkan Klaster (AI)", x_axis_label="Tahun", y_axis_label="Energi (TWh)", width=850, height=450, tools="pan,box_zoom,reset,hover,save")
    for i, cluster in enumerate(clusters):
        cluster_data = energy_by_year_cluster[energy_by_year_cluster["Cluster"] == cluster]
        source = ColumnDataSource(cluster_data)
        p9.line(x='Year', y='Total Energy Consumption (TWh)', source=source, line_width=2, color=colors[i], legend_label=f"Cluster {cluster}")
        p9.circle(x='Year', y='Total Energy Consumption (TWh)', source=source, size=5, color=colors[i], fill_alpha=0.7)
    hover = HoverTool(tooltips=[("Tahun", "@Year"), ("Klaster", "@Cluster"), ("Energi", "@{Total Energy Consumption (TWh)}{0.0} TWh")])
    p9.add_tools(hover)
    p9.legend.title = "Klaster AI"
    p9.legend.location = "top_left"
    p9.legend.click_policy = "hide"
    p9.title.align = 'center'
    st.bokeh_chart(p9, use_container_width=True)

    # 2. Top 10 negara konsumsi energi berdasarkan klaster
    st.subheader("Top 10 Negara dengan Konsumsi Energi Tertinggi Berdasarkan Klaster AI")
    st.info("""
    Visualisasi ini menampilkan 10 negara dengan konsumsi energi tertinggi dan klaster AI tempat mereka tergolong. Setiap batang warna mewakili klaster hasil dari model Agglomerative Clustering. Klaster membantu mengelompokkan negara berdasarkan karakteristik seperti konsumsi per kapita, ketergantungan bahan bakar fosil, dan emisi karbon. Dengan ini, kita dapat melihat bahwa negara-negara dengan konsumsi energi tinggi tidak selalu berada di klaster yang sama — menunjukkan adanya perbedaan signifikan dalam pola penggunaan energi.
    """)
    country_avg_ai = df_agglo.groupby(["Country", "Cluster"])["Total Energy Consumption (TWh)"].mean().reset_index()
    top10_ai = country_avg_ai.sort_values(by="Total Energy Consumption (TWh)", ascending=False).head(10)
    cluster_colors = {str(i): Category10[3][i] for i in range(3)}
    top10_ai["color"] = top10_ai["Cluster"].map(cluster_colors)
    source10 = ColumnDataSource(top10_ai)
    p10 = figure(x_range=top10_ai["Country"], title="Top 10 Negara dengan Konsumsi Energi Tertinggi Berdasarkan Klaster AI", x_axis_label='Negara', y_axis_label='Rata-rata Konsumsi Energi (TWh)', width=850, height=400, tools="pan,box_zoom,reset,hover,save")
    p10.vbar(x='Country', top='Total Energy Consumption (TWh)', source=source10, width=0.6, fill_color='color', legend_field='Cluster')
    p10.add_tools(HoverTool(tooltips=[("Negara", "@Country"), ("Energi", "@{Total Energy Consumption (TWh)}{0.0} TWh"), ("Klaster", "@Cluster")]))
    p10.xaxis.major_label_orientation = 1.0
    p10.title.align = 'center'
    p10.title.text_font_size = '14pt'
    p10.legend.title = "Klaster AI"
    p10.legend.location = "top_right"
    p10.legend.click_policy = "hide"
    st.bokeh_chart(p10, use_container_width=True)

    # 3. Rata-rata emisi karbon per tahun berdasarkan klaster
    st.subheader("Rata-Rata Emisi Karbon Per Tahun Berdasarkan Klaster AI")
    st.info("""
    Visualisasi ini menampilkan tren rata-rata emisi karbon global per tahun untuk masing-masing klaster hasil model AI. Klaster yang cenderung memiliki emisi lebih tinggi menunjukkan karakteristik negara dengan konsumsi energi fosil dominan. Sebaliknya, klaster dengan tren penurunan atau emisi rendah dapat diindikasikan sebagai negara-negara yang mulai transisi ke energi bersih atau efisiensi tinggi. Tren ini membantu memahami peran klaster dalam kontribusi terhadap emisi karbon global dari waktu ke waktu.
    """)
    emission_by_year_cluster = df_agglo.groupby(["Year", "Cluster"])["Carbon Emissions (Million Tons)"].mean().reset_index()
    p11 = figure(title="Rata-Rata Emisi Karbon Per Tahun Berdasarkan Klaster AI", x_axis_label="Tahun", y_axis_label="Emisi Karbon (Juta Ton)", width=850, height=450, tools="pan,box_zoom,reset,hover,save")
    for i, cluster in enumerate(clusters):
        cluster_data = emission_by_year_cluster[emission_by_year_cluster["Cluster"] == cluster]
        source = ColumnDataSource(cluster_data)
        p11.line(x='Year', y='Carbon Emissions (Million Tons)', source=source, line_width=2, color=colors[i], legend_label=f"Cluster {cluster}")
        p11.circle(x='Year', y='Carbon Emissions (Million Tons)', source=source, size=5, color=colors[i], fill_alpha=0.7)
    hover2 = HoverTool(tooltips=[("Tahun", "@Year"), ("Klaster", "@Cluster"), ("Emisi", "@{Carbon Emissions (Million Tons)}{0.0} Juta Ton")])
    p11.add_tools(hover2)
    p11.legend.title = "Klaster AI"
    p11.legend.location = "top_left"
    p11.legend.click_policy = "hide"
    p11.title.align = 'center'
    p11.title.text_font_size = '14pt'
    st.bokeh_chart(p11, use_container_width=True)

    # 4. Bar chart proporsi energi terbarukan per klaster
    st.subheader("Rata-Rata Proporsi Energi Terbarukan Per Klaster (AI)")
    st.info("""
    Grafik ini menggambarkan rata-rata proporsi energi terbarukan pada setiap klaster hasil model AI. Klaster dengan proporsi tertinggi mengindikasikan negara-negara yang telah beralih ke energi terbarukan dalam skala besar, sementara klaster dengan nilai lebih rendah mungkin masih bergantung pada energi fosil. Dengan pendekatan ini, kita dapat membandingkan tingkat adopsi energi terbarukan antar kelompok negara secara sistematis.
    """)
    renew_per_cluster = df_agglo.groupby("Cluster")["Renewable Energy Share (%)"].mean().reset_index()
    renew_per_cluster["Color"] = Category10[len(renew_per_cluster)]
    source_bar = ColumnDataSource(renew_per_cluster)
    p_bar = figure(y_range=renew_per_cluster["Cluster"].astype(str), width=700, height=400, title="Rata-Rata Proporsi Energi Terbarukan per Klaster (AI)", x_axis_label='Proporsi Energi Terbarukan (%)', tools="pan,box_zoom,reset,hover,save")
    p_bar.hbar(y='Cluster', right='Renewable Energy Share (%)', height=0.5, source=source_bar, fill_color='Color')
    p_bar.add_tools(HoverTool(tooltips=[("Klaster", "@Cluster"), ("Proporsi Terbarukan", "@{Renewable Energy Share (%)}{0.0}%")]))
    p_bar.title.align = 'center'
    st.bokeh_chart(p_bar, use_container_width=True)

    # 5. Area chart tren emisi karbon per tahun untuk klaster yang dipilih
    st.subheader("Area Chart: Rata-Rata Emisi Karbon Global Per Tahun Berdasarkan Klaster (AI)")
    st.info("""
    Grafik area ini menampilkan rata-rata emisi karbon global dari tahun ke tahun berdasarkan hasil pengelompokan klaster AI. Setiap klaster menunjukkan tren emisi karbon yang berbeda. Beberapa klaster cenderung stabil, sedangkan yang lain mengalami peningkatan atau penurunan drastis. Perbedaan ini mengindikasikan adanya karakteristik unik dalam konsumsi energi dan kebijakan lingkungan di masing-masing kelompok negara.
    """)
    clusters_area = sorted(df_agglo["Cluster"].unique())
    cluster_selected = st.selectbox("Pilih Klaster untuk Area Chart", clusters_area, key="area_chart_cluster")
    carbon_cluster = df_agglo[df_agglo["Cluster"] == cluster_selected].groupby(["Year", "Cluster"])["Carbon Emissions (Million Tons)"].mean().reset_index()
    color_area = Category10[len(clusters_area)][clusters_area.index(cluster_selected)]
    p_area_selected = figure(
        title=f"Rata-Rata Emisi Karbon Global Per Tahun - Klaster {cluster_selected}",
        x_axis_label="Tahun",
        y_axis_label="Emisi Karbon (Juta Ton)",
        width=850,
        height=450,
        tools="pan,box_zoom,reset,hover,save"
    )
    source = ColumnDataSource(carbon_cluster)
    p_area_selected.varea(
        x='Year',
        y1=0,
        y2='Carbon Emissions (Million Tons)',
        source=source,
        fill_color=color_area,
        fill_alpha=0.3,
        legend_label=f"Cluster {cluster_selected}"
    )
    p_area_selected.line(
        x='Year',
        y='Carbon Emissions (Million Tons)',
        source=source,
        line_width=2,
        color=color_area
    )
    p_area_selected.add_tools(HoverTool(tooltips=[
        ("Tahun", "@Year"),
        ("Klaster", "@Cluster"),
        ("Emisi", "@{Carbon Emissions (Million Tons)}{0.0} Juta Ton")
    ]))
    p_area_selected.legend.title = "Klaster AI"
    p_area_selected.legend.location = "top_left"
    p_area_selected.legend.click_policy = "hide"
    p_area_selected.title.align = 'center'
    st.bokeh_chart(p_area_selected, use_container_width=True)

    # 6. Donut chart komposisi energi per klaster (interaktif)
    st.subheader("Komposisi Sumber Energi Per Klaster (AI)")
    st.info("""
    Visualisasi ini menunjukkan bagaimana komposisi rata-rata sumber energi (terbarukan, fosil, lainnya) berbeda-beda di setiap klaster hasil model AI. Klaster dengan dominasi energi terbarukan menunjukkan proporsi energi ramah lingkungan yang lebih besar dan kemungkinan strategi energi berkelanjutan. Klaster dengan dominasi bahan bakar fosil umumnya menghasilkan emisi karbon lebih tinggi. Perbedaan ini mencerminkan keberagaman strategi dan kemampuan negara-negara dalam transisi energi.
    """)
    klaster_pilihan = st.selectbox("Pilih Klaster untuk Pie Chart", sorted(df_agglo["Cluster"].unique()), key="donut_klaster")
    df_klaster = df_agglo[df_agglo["Cluster"] == klaster_pilihan]
    renew = df_klaster["Renewable Energy Share (%)"].mean()
    fossil = df_klaster["Fossil Fuel Dependency (%)"].mean()
    other = 100 - (renew + fossil)
    data = pd.Series({'Energi Terbarukan': renew, 'Bahan Bakar Fosil': fossil, 'Lainnya': other}).reset_index(name='value').rename(columns={'index': 'sumber'})
    data['angle'] = data['value'] / data['value'].sum() * 2 * pi
    data['color'] = Category20c[len(data)]
    source_donut = ColumnDataSource(data)
    p_donut = figure(height=600, width=400, title=f"Komposisi Sumber Energi - Klaster {klaster_pilihan}", toolbar_location=None, tools="hover", tooltips="@sumber: @value{0.2f}%", x_range=(-0.5, 1.0))
    p_donut.wedge(x=0, y=1, radius=0.4, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'), line_color="white", fill_color='color', legend_field='sumber', source=source_donut)
    p_donut.annular_wedge(x=0, y=1, inner_radius=0.2, outer_radius=0.4, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'), fill_color='color', line_color="white", source=source_donut)
    p_donut.axis.visible = False
    p_donut.grid.visible = False
    p_donut.title.align = 'center'
    st.bokeh_chart(p_donut)

    # 7. Heatmap korelasi fitur per klaster (interaktif)
    st.subheader("Heatmap Korelasi Fitur Energi Per Klaster (AI)")
    st.info("""
    Visualisasi ini memperlihatkan hubungan antar fitur-fitur energi utama dalam bentuk matriks korelasi, yang dipisahkan berdasarkan hasil klaster dari model AI. Setiap heatmap mewakili satu klaster, menampilkan sejauh mana dua fitur saling berkorelasi — baik positif maupun negatif. Perbedaan pola korelasi antar klaster ini menegaskan bahwa masing-masing kelompok negara memiliki profil energi yang khas, baik dari segi struktur konsumsi, harga energi, maupun kontribusi terhadap emisi. Visualisasi ini memberikan wawasan penting bagi pembuat kebijakan untuk merancang strategi energi yang disesuaikan dengan karakteristik klaster masing-masing.
    """)
    klaster_heatmap = st.selectbox("Pilih Klaster untuk Heatmap", sorted(df_agglo["Cluster"].unique()), key="heatmap_klaster")
    df_heatmap = df_agglo[df_agglo["Cluster"] == klaster_heatmap]
    fitur_heatmap = [
        "Total Energy Consumption (TWh)",
        "Per Capita Energy Use (kWh)",
        "Renewable Energy Share (%)",
        "Fossil Fuel Dependency (%)",
        "Carbon Emissions (Million Tons)",
        "Energy Price Index (USD/kWh)"
    ]
    corr_matrix_klaster = df_heatmap[fitur_heatmap].corr().round(2)
    corr_df_klaster = corr_matrix_klaster.stack().reset_index()
    corr_df_klaster.columns = ["Feature_X", "Feature_Y", "Correlation"]
    source_heatmap = ColumnDataSource(corr_df_klaster)
    mapper_heatmap = LinearColorMapper(palette=Viridis256, low=-1, high=1)
    p_heatmap = figure(title=f"Heatmap Korelasi Fitur Energi - Klaster {klaster_heatmap}", x_range=fitur_heatmap, y_range=list(reversed(fitur_heatmap)), x_axis_location="above", width=800, height=600, tools="hover,save", tooltips=[('X', '@Feature_X'), ('Y', '@Feature_Y'), ('Korelasi', '@Correlation')])
    p_heatmap.rect(x="Feature_X", y="Feature_Y", width=1, height=1, source=source_heatmap, fill_color=linear_cmap('Correlation', Viridis256, -1, 1), line_color=None)
    color_bar_heatmap = ColorBar(color_mapper=mapper_heatmap, ticker=BasicTicker(desired_num_ticks=10), formatter=PrintfTickFormatter(format="%.2f"), label_standoff=12, border_line_color=None, location=(0, 0))
    p_heatmap.add_layout(color_bar_heatmap, 'right')
    p_heatmap.xaxis.major_label_orientation = np.pi / 4
    p_heatmap.title.align = 'center'
    st.bokeh_chart(p_heatmap, use_container_width=True)

    # 8. Scatter plot emisi karbon vs energi terbarukan per klaster (interaktif)
    st.subheader("Scatter Plot: Emisi Karbon vs Energi Terbarukan Per Klaster")
    st.info("""
    Visualisasi ini menampilkan hubungan antara emisi karbon dan proporsi energi terbarukan untuk negara-negara di seluruh dunia, yang telah dikelompokkan berdasarkan klaster hasil model AI. Setiap titik pada scatter plot mewakili satu negara, dan warna menunjukkan klaster AI tempat negara tersebut berada. Visualisasi ini memungkinkan identifikasi perbedaan pola hubungan antar fitur utama energi. Klaster tertentu memperlihatkan negara-negara dengan emisi karbon tinggi dan proporsi energi terbarukan yang rendah, mencerminkan ketergantungan kuat pada energi berbasis fosil. Klaster lainnya menunjukkan negara-negara dengan proporsi energi terbarukan tinggi dan emisi yang lebih rendah, menandakan pendekatan yang lebih bersih dan berkelanjutan terhadap penggunaan energi. Distribusi titik-titik pada masing-masing klaster menggambarkan variasi strategi energi dan efektivitas kebijakan lingkungan yang diambil oleh kelompok negara tersebut. Visualisasi ini memberikan wawasan penting untuk analisis perbandingan lintas negara dan menyusun kebijakan berbasis kelompok dengan karakteristik serupa.
    """)
    klaster_pilihan2 = st.selectbox("Pilih Klaster untuk Scatter Plot", sorted(df_agglo["Cluster"].unique()), key="scatter_klaster")
    df_klaster2 = df_agglo[df_agglo["Cluster"] == klaster_pilihan2]
    avg_per_country = df_klaster2.groupby("Country")[["Carbon Emissions (Million Tons)", "Renewable Energy Share (%)"]].mean().reset_index()
    source_scatter = ColumnDataSource(avg_per_country)
    p_scatter = figure(title=f"Scatter: Emisi Karbon vs Energi Terbarukan (Klaster {klaster_pilihan2})", x_axis_label="Proporsi Energi Terbarukan (%)", y_axis_label="Emisi Karbon (Juta Ton)", width=850, height=500, tools="pan,box_zoom,reset,hover,save")
    p_scatter.circle(x='Renewable Energy Share (%)', y='Carbon Emissions (Million Tons)', size=10, source=source_scatter, fill_color="navy", line_color="black", fill_alpha=0.7)
    p_scatter.add_tools(HoverTool(tooltips=[("Negara", "@Country"), ("Energi Terbarukan", "@{Renewable Energy Share (%)}{0.0}%"), ("Emisi Karbon", "@{Carbon Emissions (Million Tons)}{0.0} Juta Ton")]))
    p_scatter.title.align = 'center'
    st.bokeh_chart(p_scatter, use_container_width=True) 