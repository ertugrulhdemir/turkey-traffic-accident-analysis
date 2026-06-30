import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import pycountry
import numpy as np
import requests
import locale
import re
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Turkey Traffic Accident Analysis", page_icon="🚗", layout="wide")

def tr_sort_key(s):
    replacements = [
        ('ç','c'),('Ç','C'),
        ('ğ','g'),('Ğ','G'),
        ('ı','i'),('İ','I'),
        ('ö','o'),('Ö','O'),
        ('ş','s'),('Ş','S'),
        ('ü','u'),('Ü','U'),
    ]
    result = s
    for old, new in replacements:
        result = result.replace(old, new)
    return result.lower()

def fmt(n):
    """Format large numbers as 1.2M, 45K etc."""
    try:
        n = int(round(float(n)))
    except:
        return str(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 16px; }
    .tab-desc {
        background: #1e2433;
        border-left: 4px solid #4a9eff;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        color: #cbd5e0;
        font-size: 0.95rem;
    }
    [data-testid="stDataFrame"] iframe { pointer-events: none; }
</style>
""", unsafe_allow_html=True)

C_BLUE   = '#4a9eff'
C_RED    = '#e05252'
C_ORANGE = '#f6a84b'
C_GREEN  = '#52c97a'
C_TEAL   = '#38b2ac'

def df_with_excel_download(df, filename, key):
    """Render a dataframe plus a TR-Excel-compatible CSV download button.
    Excel on Turkish-locale Windows expects ';' as the column separator
    (since ',' is the decimal separator), and needs a UTF-8 BOM to render
    Turkish characters (İ, ş, ğ, ç, ö, ü) correctly. The native Streamlit
    download button uses ',' which causes Excel to dump the whole row into
    cell A1 instead of splitting it into columns.

    Decimal numbers (e.g. 31.3) are written with a comma decimal separator
    (31,3) instead of a dot. Without this, Excel's TR locale tries to parse
    dotted decimals like '31.3' or '30.5' as dates (31 Mart, 30 Mayıs) since
    they look like valid day.month patterns.
    """
    st.dataframe(df, use_container_width=True, hide_index=True)
    export_df = df.copy()
    for c in export_df.select_dtypes(include='float').columns:
        export_df[c] = export_df[c].map(lambda v: str(v).replace('.', ',') if pd.notna(v) else v)
    csv_bytes = export_df.to_csv(index=False, sep=';').encode('utf-8-sig')
    st.download_button(
        label="📥 Excel için indir (TR uyumlu)",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key
    )
    st.caption("ℹ️ Tablonun sağ üstündeki ⤓ ikonuyla indirirseniz Excel'de (TR) tüm satır tek hücreye sıkışabilir. Doğru görüntü için yukarıdaki butonu kullanın.")


# ── Data loading functions ──────────────────────────────────────────────────

@st.cache_data
def load_province_data(file, year):
    try:
        raw = pd.read_excel(file, header=None)

        if year in [2013, 2014]:
            start_row = None
            for i in range(raw.shape[0]):
                if 'Adana' in str(raw.iloc[i, 0]):
                    start_row = i
                    break
            if start_row is None:
                return pd.DataFrame()
            left  = raw.iloc[start_row:, [0, 1, 2, 3]].copy()
            right = raw.iloc[start_row:, [14, 15, 16, 17]].copy()

        elif year == 2016:
            start_row = 7
            left  = raw.iloc[start_row:, [0, 1, 5, 8]].copy()
            right = raw.iloc[start_row:, [11, 12, 16, 19]].copy()

        elif year in [2015, 2017, 2018, 2019]:
            start_row = 7
            left  = raw.iloc[start_row:, [0, 1, 4, 7]].copy()
            right = raw.iloc[start_row:, [10, 11, 14, 17]].copy()

        elif year in [2020, 2021, 2022, 2023]:
            start_row = 6
            left  = raw.iloc[start_row:, [0, 1, 4, 7]].copy()
            right = raw.iloc[start_row:, [10, 11, 14, 17]].copy()

        elif year in [2024, 2025]:
            start_row = 4
            left  = raw.iloc[start_row:, [0, 1, 8, 11]].copy()
            right = raw.iloc[start_row:, [14, 15, 20, 21]].copy()

        else:
            return pd.DataFrame()

        cols = ['province', 'total_accidents', 'deaths', 'injuries']
        left.columns  = cols
        right.columns = cols

        df = pd.concat([left, right], ignore_index=True)
        df = df[df['province'].notna()]
        df = df[~df['province'].astype(str).str.strip().isin(['', 'Toplam-Total', 'nan'])]
        for col in ['total_accidents', 'deaths', 'injuries']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        # Normalize province names for consistency
        province_name_fixes = {
        'Ş.Urfa': 'Şanlıurfa',
        'Afyon': 'Afyonkarahisar',
        'İçel': 'Mersin',
        'K.Maraş': 'Kahramanmaraş',
        'Kilis ': 'Kilis',  # trailing space
        }
        df['province'] = df['province'].str.strip()
        df['province'] = df['province'].replace(province_name_fixes)
        # Keep one row per province (in case of duplicates across left/right blocks)
        df = df.groupby('province').agg({
            'total_accidents': 'sum',
            'deaths': 'sum',
            'injuries': 'sum'
        }).reset_index()
        df['year'] = year
        df['deaths_per_1000'] = (df['deaths'] / df['total_accidents'] * 1000).round(2)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_vehicle_data():
    """Load registered vehicle count by year (2010-2025)."""
    try:
        raw = pd.read_excel('registered_vehicles_2025.xls', header=None)
        df = raw.iloc[10:26, [0, 1]].copy()
        df.columns = ['year', 'registered_vehicles']
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['registered_vehicles'] = pd.to_numeric(df['registered_vehicles'], errors='coerce')
        return df.dropna().reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_risk_data():
    """Load normalized risk metrics per 100k population and 100k vehicles (2015-2025)."""
    try:
        raw = pd.read_excel('risk_normalized_2025.xls', header=None)
        df = raw.iloc[9:20, [0, 1, 2, 4, 5]].copy()
        df.columns = ['year', 'deaths_per_100k_pop', 'injuries_per_100k_pop',
                      'deaths_per_100k_veh', 'injuries_per_100k_veh']
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        for col in ['deaths_per_100k_pop', 'injuries_per_100k_pop',
                    'deaths_per_100k_veh', 'injuries_per_100k_veh']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna().reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_fault_yearly():
    """Load fault category totals for 2020-2025 for use as ML features."""
    records = []
    for year in range(2020, 2026):
        try:
            df = pd.read_excel(f'fault_{year}.xls', engine='xlrd', header=None)
            row = {'year': year}
            for i, r in df.iterrows():
                v0 = str(r.iloc[0]).strip()
                v2 = pd.to_numeric(str(r.iloc[2]).strip(), errors='coerce')
                if 'Toplam - Total' in v0:
                    row['fault_total'] = v2
                elif 'Sürücü kusurları' in v0:
                    row['driver_faults'] = v2
                elif 'Yolcu kusurları' in v0:
                    row['passenger_faults'] = v2
                elif 'Yaya kusurları' in v0:
                    row['pedestrian_faults'] = v2
                elif 'Yol kusurları' in v0:
                    row['road_faults'] = v2
                elif 'Taşıt kusurları' in v0:
                    row['vehicle_faults'] = v2
            if 'fault_total' in row and row['fault_total'] > 0:
                row['driver_fault_ratio'] = row.get('driver_faults', 0) / row['fault_total']
                row['pedestrian_fault_ratio'] = row.get('pedestrian_faults', 0) / row['fault_total']
                row['road_fault_ratio'] = row.get('road_faults', 0) / row['fault_total']
            records.append(row)
        except Exception:
            pass
    return pd.DataFrame(records)

@st.cache_data(ttl=86400)
def load_who_data():
    """Fetch road traffic death rates for all countries from WHO API."""
    r = requests.get(
        'https://ghoapi.azureedge.net/api/RS_198?%24orderby=TimeDim%20desc',
        timeout=10
    )
    data = r.json()
    return pd.DataFrame(data['value'])

def get_country_name(iso3):
    """Convert ISO3 country code to full country name."""
    try:
        return pycountry.countries.get(alpha_3=iso3).name
    except Exception:
        return iso3

def blend_forecast(X_tr, y_tr, X_fc):
    """
    Blend GradientBoosting + Ridge regression for smoother extrapolation.
    GradientBoosting captures non-linear patterns; Ridge handles trend extrapolation.
    Near-term forecasts weight GB more; far-term forecasts weight Ridge more.
    """
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=3, random_state=42)
    gb.fit(X_tr, y_tr)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    gb_pred    = gb.predict(X_fc)
    ridge_pred = ridge.predict(X_fc)
    weights = [0.6, 0.55, 0.5, 0.45, 0.4]  # ridge weight per forecast year
    blended = [w * r + (1 - w) * g for w, r, g in zip(weights, ridge_pred, gb_pred)]
    fi = gb.feature_importances_ if hasattr(gb, 'feature_importances_') else None
    return np.array(blended), fi

@st.cache_data
def load_fault_data():
    try:
        raw = pd.read_excel('fault_2025.xls', header=None)
        df = raw.iloc[5:21, [0, 1, 4, 7, 10, 13, 16]].copy()
        df.columns = ['year', 'total', 'driver', 'passenger', 'pedestrian', 'road', 'vehicle']
        for col in ['total', 'driver', 'passenger', 'pedestrian', 'road', 'vehicle']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        return df.dropna().reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
    
@st.cache_data(ttl=86400)
def check_tuik_latest():
    url = 'https://veriportali.tuik.gov.tr/api/tr/data/search'
    payload = {
        "text": "Karayolu Trafik Kaza İstatistikleri",
        "page": 1,
        "typeIds": [1],
        "categoryIds": [],
        "subCategoryIds": [],
        "years": [],
        "levels": [],
        "archive": False,
        "autoFilter": False
    }
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://veriportali.tuik.gov.tr",
        "Referer": "https://veriportali.tuik.gov.tr/tr/search?q=Karayolu%20Trafik%20Kaza%20%C4%B0statistikleri",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=5)
        data = r.json()
        for item in data['data']['data']:
            if item.get('type') == 1 and 'Karayolu Trafik Kaza İstatistikleri' in item.get('title', ''):
                match = re.search(r'(\d{4})', item['title'])
                if match:
                    return int(match.group(1))
    except Exception:
        pass
    return None

@st.cache_data
def load_vehicle_accident_data():
    try:
        results = []
        for year in range(2020, 2026):
            f = f'vehicle_accident_{year}.xls'
            raw = pd.read_excel(f, header=None, engine='xlrd')
            row = raw.iloc[8]
            if year in [2020, 2021]:
                involved = pd.to_numeric(row.iloc[2], errors='coerce')
            else:
                involved = pd.to_numeric(row.iloc[1], errors='coerce')
            if pd.notna(involved):
                results.append({'year': year, 'vehicles_involved': involved})
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_monthly_data():
    try:
        raw = pd.read_excel('monthly_2025.xls', header=None)
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        df = raw.iloc[6:18, [0, 1, 3, 8]].copy()
        df.columns = ['month_raw', 'accidents', 'deaths', 'injuries']
        df['month'] = months
        for col in ['accidents', 'deaths', 'injuries']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df[['month', 'accidents', 'deaths', 'injuries']].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_daily_data():
    try:
        return pd.read_excel('daily_2025.xls', header=None)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_all_years():
    frames = []
    for y in range(2013, 2026):
        df = load_province_data(f'accidents_province_{y}.xls', y)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Load data ───────────────────────────────────────────────────────────────
df_all   = load_all_years()
df_2025  = df_all[df_all['year'] == 2025].copy()

yearly = df_all.groupby('year').agg(
    total_accidents=('total_accidents', 'sum'),
    deaths=('deaths', 'sum'),
    injuries=('injuries', 'sum')
).reset_index()

# Merge vehicle data into yearly (2013-2025 intersection)
vehicle_df = load_vehicle_data()
if not vehicle_df.empty:
    yearly = yearly.merge(vehicle_df, on='year', how='left')
else:
    yearly['registered_vehicles'] = np.nan
# Merge vehicles-involved counts into yearly (2020-2025 only)
veh_acc_df = load_vehicle_accident_data()
if not veh_acc_df.empty:
    yearly = yearly.merge(veh_acc_df, on='year', how='left')

# Merge fault ratios into yearly (2020-2025)
fault_yearly_df = load_fault_yearly()
if not fault_yearly_df.empty:
    yearly = yearly.merge(
        fault_yearly_df[['year', 'driver_fault_ratio', 'pedestrian_fault_ratio', 'road_fault_ratio']],
        on='year', how='left'
    )
else:
    yearly['driver_fault_ratio'] = np.nan

# ── Header ──────────────────────────────────────────────────────────────────
st.title("Turkey Traffic Accident Analysis")
st.markdown("**Source:** TÜİK Road Traffic Accident Statistics 2013–2025 · 81 Provinces")
st.divider()

# ── Navigation (session_state-driven, fake-tab) ──────────────────────────────
TAB_LABELS = [
    "Home", "13-Year Trend", "Province Analysis", "Time Analysis",
    "Fault Analysis", "ML Prediction", "Map", "Global Context"
]

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Home"

def _on_nav_change():
    st.session_state.active_tab = st.session_state.nav_radio

def go_to(label):
    """Switch the active tab from a button click and re-run."""
    st.session_state.active_tab = label
    st.rerun()

# The tab strip only shows once you're past Home — Home itself shows only the card grid.
if st.session_state.active_tab != "Home":
    st.radio(
        "Navigation", TAB_LABELS,
        index=TAB_LABELS.index(st.session_state.active_tab),
        key='nav_radio',
        horizontal=True, label_visibility='collapsed',
        on_change=_on_nav_change
    )
    st.divider()

match st.session_state.active_tab:

    case "Home":
        _deadliest_month_home = load_monthly_data().sort_values('deaths', ascending=False).iloc[0]['month']
        _most_dangerous_prov  = df_2025.loc[df_2025['deaths_per_1000'].idxmax(), 'province']
        _most_accidents_prov  = df_2025.loc[df_2025['total_accidents'].idxmax(), 'province']
        _fault_df_home        = load_fault_data()
        _driver_fault_pct     = None
        if not _fault_df_home.empty:
            _last_fault = _fault_df_home[_fault_df_home['year'] == 2025].iloc[0]
            _fault_total = _last_fault['driver'] + _last_fault['passenger'] + _last_fault['pedestrian'] + _last_fault['road'] + _last_fault['vehicle']
            _driver_fault_pct = _last_fault['driver'] / _fault_total * 100

        st.markdown(f"""
        <p style="font-size: 16px; color: #cbd5e0; line-height: 1.7; max-width: 640px;">
        TÜİK'in 2013–2025 arası resmi karayolu trafik kaza verilerini 81 il bazında inceleyen
        interaktif bir analiz panosu — nerede, ne zaman ve neden kaza oluyor, önümüzdeki 5 yıl nasıl görünüyor.
        </p>
        """, unsafe_allow_html=True)
        st.write("")

        cards = [
            ("13-Year Trend", f"{int(round(df_2025['total_accidents'].sum())):,}".replace(',', '.') +
             f" kaza ile 2025, COVID sonrası en yüksek seviyede."),
            ("Province Analysis", f"{_most_accidents_prov} en çok kaza, {_most_dangerous_prov} kaza başına en yüksek ölüm oranına sahip."),
            ("Time Analysis", f"{_deadliest_month_home}, en ölümcül ay olarak öne çıkıyor."),
            ("Fault Analysis",
             (f"Kazaların %{_driver_fault_pct:.0f}'i sürücü kusurundan kaynaklanıyor." if _driver_fault_pct else "Kaza nedenleri kategori bazında inceleniyor.")),
            ("ML Prediction", "2030'a kadar kaza sayısının artmaya devam etmesi bekleniyor."),
            ("Map", "81 ili interaktif harita üzerinde karşılaştırın."),
            ("Global Context", "Türkiye'nin ölüm oranı dünya ortalamasıyla kıyaslanıyor."),
        ]

        row1 = st.columns(4)
        for i in range(4):
            label, insight = cards[i]
            with row1[i]:
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.caption(insight)
                    if st.button("Aç →", key=f'home_btn_{i}', use_container_width=True):
                        go_to(label)

        row2 = st.columns(3)
        for i in range(4, 7):
            label, insight = cards[i]
            with row2[i - 4]:
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.caption(insight)
                    if st.button("Aç →", key=f'home_btn_{i}', use_container_width=True):
                        go_to(label)

        st.caption("Bir karta tıklayarak ilgili analiz sekmesine geçin.")

    # ── 13-Year Trend ─────────────────────────────────────────────────────────
    case "13-Year Trend":
        st.markdown('<div class="tab-desc">How have road traffic accidents changed in Turkey over the last 13 years?</div>', unsafe_allow_html=True)

        deadliest_month = load_monthly_data().sort_values('deaths', ascending=False).iloc[0]['month']
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("2025 Total Accidents", f"{df_2025['total_accidents'].sum():,.0f}")
        c2.metric("2025 Total Deaths",    f"{df_2025['deaths'].sum():,.0f}")
        c3.metric("2025 Total Injuries",  f"{df_2025['injuries'].sum():,.0f}")
        c4.metric("Most Dangerous Province", df_2025.loc[df_2025['deaths_per_1000'].idxmax(), 'province'])
        c5.metric("Deadliest Month", deadliest_month)
        st.divider()   

        metric = st.radio("Show:", ["Total Accidents", "Deaths", "Injuries"], horizontal=True)
        col_map = {'Total Accidents': 'total_accidents', 'Deaths': 'deaths', 'Injuries': 'injuries'}
        col = col_map[metric]

        fig = px.line(yearly, x='year', y=col, markers=True,
                      title=f'Turkey — {metric} (2013–2025)',
                      labels={'year': 'Year', col: metric})
        fig.update_traces(line_color='steelblue', marker_size=8)
        fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor='red', opacity=0.1,
                      annotation_text="COVID-19", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Province-Level Annual Trend")
        selected_province = st.selectbox("Select a province:", sorted(df_all['province'].unique(), key=tr_sort_key))
        prov_trend = df_all[df_all['province'] == selected_province]
        fig2 = px.bar(prov_trend, x='year', y='total_accidents', color='deaths',
                      title=f'{selected_province} — Annual Accidents & Deaths',
                      labels={'total_accidents': 'Total Accidents', 'year': 'Year', 'deaths': 'Deaths'},
                      color_continuous_scale='Reds')
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        df_with_excel_download(
            df_2025[['province','total_accidents','deaths','injuries','deaths_per_1000']]
            .sort_values('total_accidents', ascending=False)
            .rename(columns={'province':'Province','total_accidents':'Total Accidents',
                             'deaths':'Deaths','injuries':'Injuries',
                             'deaths_per_1000':'Deaths/1000 Acc.'}),
            filename='turkey_provinces_2025.csv', key='dl_tab1'
        )
        st.caption("Data: TÜİK — veriportali.tuik.gov.tr | Developer: Ertuğrul Halisdemir")

    # ── TAB 2: Province Analysis ─────────────────────────────────────────────────
    case "Province Analysis":
        st.markdown('<div class="tab-desc">Which provinces have the most accidents — and which are the deadliest?</div>', unsafe_allow_html=True)
        col_left, col_right = st.columns([1, 2])

        with col_left:
            prov_sel = st.selectbox("Select a province:", sorted(df_2025['province'].tolist(), key=tr_sort_key), key='prov2')
            prov_data = df_2025[df_2025['province'] == prov_sel].iloc[0]
            st.metric("Total Accidents",      f"{prov_data['total_accidents']:,.0f}")
            st.metric("Deaths",               f"{prov_data['deaths']:,.0f}")
            st.metric("Injuries",             f"{prov_data['injuries']:,.0f}")
            st.metric("Deaths per 1000 Acc.", f"{prov_data['deaths_per_1000']}")
            acc_rank  = int(df_2025['total_accidents'].rank(ascending=False).loc[df_2025['province'] == prov_sel].values[0])
            risk_rank = int(df_2025['deaths_per_1000'].rank(ascending=False).loc[df_2025['province'] == prov_sel].values[0])
            st.info(f"Ranked **#{acc_rank}** in accident count\nRanked **#{risk_rank}** in danger rate")

        with col_right:
            view = st.radio("", ["Most Accidents", "Most Dangerous", "Accidents vs Deaths"], horizontal=True, key='r2')
            if view == "Most Accidents":
                top15 = df_2025.sort_values('total_accidents', ascending=False).head(15)
                fig = px.bar(top15, x='total_accidents', y='province', orientation='h',
                             color='total_accidents', color_continuous_scale='Blues',
                             title='Top 15 Provinces by Accident Count (2025)',
                             labels={'total_accidents': 'Total Accidents', 'province': ''})
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            elif view == "Most Dangerous":
                top15d = df_2025.sort_values('deaths_per_1000', ascending=False).head(15)
                fig = px.bar(top15d, x='deaths_per_1000', y='province', orientation='h',
                             color='deaths_per_1000', color_continuous_scale='Reds',
                             title='Top 15 Most Dangerous Provinces — Deaths per 1000 Accidents (2025)',
                             labels={'deaths_per_1000': 'Deaths per 1000 Acc.', 'province': ''})
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter(df_2025, x='total_accidents', y='deaths', text='province',
                                 color='deaths_per_1000', color_continuous_scale='Oranges',
                                 size='injuries', hover_name='province',
                                 title='Accidents vs Deaths by Province (2025)',
                                 labels={'total_accidents': 'Total Accidents', 'deaths': 'Deaths'})
                fig.update_traces(textposition='top center', textfont_size=8)
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
        df_with_excel_download(
            df_2025[['province','total_accidents','deaths','injuries','deaths_per_1000']]
            .sort_values('total_accidents', ascending=False)
            .rename(columns={'province':'Province','total_accidents':'Total Accidents',
                             'deaths':'Deaths','injuries':'Injuries',
                             'deaths_per_1000':'Deaths/1000 Acc.'}),
            filename='turkey_provinces_2025.csv', key='dl_tab2'
        )
        st.caption("Data: TÜİK — veriportali.tuik.gov.tr | Developer: Ertuğrul Halisdemir")

    # ── TAB 3: Time Analysis ─────────────────────────────────────────────────────
    case "Time Analysis":
        st.markdown('<div class="tab-desc">When do accidents happen most often? Monthly and daily breakdown for 2025.</div>', unsafe_allow_html=True)
    
        deadliest_month = load_monthly_data().sort_values('deaths', ascending=False).iloc[0]['month']
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("2025 Total Accidents", f"{df_2025['total_accidents'].sum():,.0f}")
        c2.metric("2025 Total Deaths",    f"{df_2025['deaths'].sum():,.0f}")
        c3.metric("2025 Total Injuries",  f"{df_2025['injuries'].sum():,.0f}")
        c4.metric("Most Dangerous Province", df_2025.loc[df_2025['deaths_per_1000'].idxmax(), 'province'])
        c5.metric("Deadliest Month", deadliest_month)
        st.divider()
    
        monthly_df = load_monthly_data()
        col_a, col_b = st.columns(2)

        with col_a:
            if not monthly_df.empty:
                fig = px.bar(monthly_df, x='month', y='accidents',
                             title='Accidents by Month (2025)',
                             labels={'month': 'Month', 'accidents': 'Accidents'},
                             color='accidents', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                fig2 = px.bar(monthly_df, x='month', y='deaths',
                              title='Deaths by Month (2025)',
                              labels={'month': 'Month', 'deaths': 'Deaths'},
                              color='deaths', color_continuous_scale='Reds')
                st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            raw_daily = load_daily_data()
            if not raw_daily.empty:
                try:
                    day_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                    tr_days   = ['Pazartesi','Salı','Çarşamba','Perşembe','Cuma','Cumartesi','Pazar']
                    day_data  = []
                    for _, row in raw_daily.iterrows():
                        v0 = str(row.iloc[0]).strip()
                        if any(d in v0 for d in tr_days + day_names):
                            vals = pd.to_numeric(row.iloc[1:], errors='coerce').dropna()
                            if len(vals) >= 1:
                                day_data.append({'day_raw': v0, 'accidents': vals.values[0]})
                    if day_data:
                        day_df = pd.DataFrame(day_data)
                        day_df['day'] = day_names[:len(day_df)]
                        fig = px.bar(day_df, x='day', y='accidents',
                                     title='Accidents by Day of Week (2025)',
                                     labels={'day': 'Day', 'accidents': 'Accidents'},
                                     color='accidents', color_continuous_scale='Purples')
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Could not process daily data: {e}")
        st.divider()
        df_with_excel_download(
            df_2025[['province','total_accidents','deaths','injuries','deaths_per_1000']]
            .sort_values('total_accidents', ascending=False)
            .rename(columns={'province':'Province','total_accidents':'Total Accidents',
                             'deaths':'Deaths','injuries':'Injuries',
                             'deaths_per_1000':'Deaths/1000 Acc.'}),
            filename='turkey_provinces_2025.csv', key='dl_tab3'
        )
        st.caption("Data: TÜİK — veriportali.tuik.gov.tr | Developer: Ertuğrul Halisdemir")

    # ── TAB 4: Fault Analysis ─────────────────────────────────────────────────────
    case "Fault Analysis":
        st.markdown('<div class="tab-desc">Who or what causes traffic accidents? Driver error, road conditions, and more.</div>', unsafe_allow_html=True)
        fault_df = load_fault_data()
        if not fault_df.empty:
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                last = fault_df[fault_df['year'] == 2025].iloc[0]
                total = last['driver'] + last['passenger'] + last['pedestrian'] + last['road'] + last['vehicle']
                pie_df = pd.DataFrame({
                    'Fault Type': [
                        f"Driver ({last['driver']/total*100:.1f}%)",
                        f"Passenger ({last['passenger']/total*100:.1f}%)",
                        f"Pedestrian ({last['pedestrian']/total*100:.1f}%)",
                        f"Road ({last['road']/total*100:.1f}%)",
                        f"Vehicle ({last['vehicle']/total*100:.1f}%)"
                    ],
                    'Count': [last['driver'], last['passenger'], last['pedestrian'], last['road'], last['vehicle']]
                })
                fig = px.pie(pie_df, values='Count', names='Fault Type',
                             title='2025 Accident Cause Breakdown',
                             color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig, use_container_width=True)
            with col_k2:
                fig2 = px.line(fault_df, x='year', y=['driver', 'pedestrian', 'road'],
                               title='Fault Type Trends (2010–2025)',
                               labels={'year': 'Year', 'value': 'Count', 'variable': 'Fault Type'},
                               markers=True)
                name_map = {'driver': 'Driver', 'pedestrian': 'Pedestrian', 'road': 'Road'}
                fig2.for_each_trace(lambda t: t.update(name=name_map.get(t.name, t.name)))
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Annual Fault Counts")
            df_with_excel_download(
                fault_df[['year','total','driver','passenger','pedestrian','road','vehicle']]
                .rename(columns={'year':'Year','total':'Total','driver':'Driver',
                                 'passenger':'Passenger','pedestrian':'Pedestrian',
                                 'road':'Road','vehicle':'Vehicle'}),
                filename='turkey_fault_analysis.csv', key='dl_tab4'
            )
        else:
            st.info("Fault data could not be loaded.")

    # ── TAB 5: ML Prediction ─────────────────────────────────────────────────────
    case "ML Prediction":
        st.markdown('<div class="tab-desc">Using machine learning, this section estimates accidents and deaths through 2030 — assuming current trends continue.</div>', unsafe_allow_html=True)

        # Build feature set — use 2020-2025 where all features available
        feature_cols = ['year', 'registered_vehicles', 'driver_fault_ratio', 'vehicles_involved']
        ml_df = yearly[yearly['year'] >= 2020].dropna(subset=feature_cols).copy()

        has_full_features = len(ml_df) >= 4
        if has_full_features:
            st.caption("Trained on 2020–2025 data using 4 features: year, registered vehicles, driver fault rate, and vehicles involved.")
        else:
            ml_df = yearly.dropna(subset=['year', 'registered_vehicles']).copy()
            st.markdown("Model trained on available data with year + registered vehicles.")
            st.warning("Some features missing — using reduced feature set.")

        # Forecast registered vehicles 2026-2030
        veh_lr = LinearRegression()
        veh_lr.fit(yearly[['year']].dropna(), yearly.dropna(subset=['registered_vehicles'])['registered_vehicles'])
        veh_inv_lr = LinearRegression()
        veh_inv_df = yearly.dropna(subset=['vehicles_involved'])
        veh_inv_lr.fit(veh_inv_df[['year']], veh_inv_df['vehicles_involved'])

        # Forecast driver fault ratio 2026-2030 (trend)
        if has_full_features:
            fault_lr = LinearRegression()
            fault_lr.fit(ml_df[['year']], ml_df['driver_fault_ratio'])

        forecast_years = list(range(2026, 2031))
        forecast_rows = []
        for fy in forecast_years:
            row = {'year': fy, 'registered_vehicles': veh_lr.predict([[fy]])[0]}
            if has_full_features:
                row['driver_fault_ratio'] = fault_lr.predict([[fy]])[0]
                row['vehicles_involved'] = veh_inv_lr.predict([[fy]])[0]
            forecast_rows.append(row)
        forecast_df = pd.DataFrame(forecast_rows)

        X_train = ml_df[feature_cols].values if has_full_features else ml_df[['year', 'registered_vehicles']].values
        X_forecast = forecast_df[feature_cols].values if has_full_features else forecast_df[['year', 'registered_vehicles']].values

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            pred_acc, fi_acc = blend_forecast(X_train, ml_df['total_accidents'].values, X_forecast)
            forecast_df['pred_accidents'] = pred_acc
            actual_2025 = yearly[yearly['year'] == 2025]['total_accidents'].values[0]
            change_acc = (pred_acc[0] - actual_2025) / actual_2025 * 100
            st.metric("2026 Predicted Accidents", fmt(pred_acc[0]), f"{change_acc:+.1f}% vs 2025")


            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['total_accidents'],
                                     mode='lines+markers', name='Actual',
                                     line=dict(color='steelblue', width=2)))
            fig.add_trace(go.Scatter(x=[2025] + forecast_years,
                                     y=[actual_2025] + list(pred_acc),
                                     mode='lines+markers', name='Forecast',
                                     line=dict(color='red', width=2, dash='dash'),
                                     marker=dict(size=8)))
            fig.update_layout(title='Total Accidents — Actual vs 2026–2030 Forecast',
                              xaxis_title='Year', yaxis_title='Accidents')
            st.plotly_chart(fig, use_container_width=True)

        with col_t2:
            # Chained death prediction
            yearly['fatal_ratio'] = yearly['deaths'] / yearly['total_accidents']
            ml_df2 = yearly[yearly['year'] >= 2020].dropna(subset=feature_cols + ['fatal_ratio']).copy()
            fatal_lr = LinearRegression()
            fatal_lr.fit(ml_df2[['year']], ml_df2['fatal_ratio'])
            forecast_fatal_ratios = fatal_lr.predict([[y] for y in forecast_years])

            pred_dth,_ = blend_forecast(X_train, ml_df['deaths'].values, X_forecast)
            chained_deaths = [acc * ratio for acc, ratio in zip(forecast_df['pred_accidents'], forecast_fatal_ratios)]
            final_deaths = [0.4 * m + 0.6 * c for m, c in zip(pred_dth, chained_deaths)]
            forecast_df['pred_deaths'] = final_deaths

            actual_2025_d = yearly[yearly['year'] == 2025]['deaths'].values[0]
            change_dth = (final_deaths[0] - actual_2025_d) / actual_2025_d * 100
            st.metric("2026 Predicted Deaths", fmt(final_deaths[0]), f"{change_dth:+.1f}% vs 2025")
            st.caption("Method: Chained prediction (accidents × fatal rate) + direct model blend")

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=yearly['year'], y=yearly['deaths'],
                                      mode='lines+markers', name='Actual',
                                      line=dict(color='darkred', width=2)))
            fig2.add_trace(go.Scatter(x=[2025] + forecast_years,
                                      y=[actual_2025_d] + final_deaths,
                                      mode='lines+markers', name='Forecast',
                                      line=dict(color='orange', width=2, dash='dash'),
                                      marker=dict(size=8)))
            fig2.update_layout(title='Deaths — Actual vs 2026–2030 Forecast (Chained)',
                               xaxis_title='Year', yaxis_title='Deaths')
            st.plotly_chart(fig2, use_container_width=True)

        # Forecast table
        st.markdown("""
        <div style="max-width: 600px; margin-bottom: 1.5rem;">
        <p style="font-size: 15px; color: var(--color-text-secondary); line-height: 1.6; margin: 0;">
        This project was built to explore 13 years of official Turkish road traffic data — 
        revealing where, when, and why accidents happen, and what the next 5 years might look like.
        </p>
        </div>
        """, unsafe_allow_html=True)


        # Normalized risk trend
        st.divider()
        st.markdown("### 📉 Normalized Risk Trend (2015–2025)")
        st.markdown("Even as total accidents rise, is Turkey getting safer *per capita* and *per vehicle*?")

        risk_df = load_risk_data()
        if not risk_df.empty:
            col_r1, col_r2 = st.columns(2)

            with col_r1:
                fig_r1 = go.Figure()
                fig_r1.add_trace(go.Scatter(x=risk_df['year'], y=risk_df['deaths_per_100k_pop'],
                                            mode='lines+markers', name='Deaths',
                                            line=dict(color='crimson', width=2)))
                fig_r1.add_trace(go.Scatter(x=risk_df['year'], y=risk_df['injuries_per_100k_pop'],
                                            mode='lines+markers', name='Injuries',
                                            line=dict(color='orange', width=2)))
                fig_r1.update_layout(title='Per 100,000 Population',
                                     xaxis_title='Year', yaxis_title='Count',
                                     legend=dict(orientation='h'))
                st.plotly_chart(fig_r1, use_container_width=True)

            with col_r2:
                fig_r2 = go.Figure()
                fig_r2.add_trace(go.Scatter(x=risk_df['year'], y=risk_df['deaths_per_100k_veh'],
                                            mode='lines+markers', name='Deaths',
                                            line=dict(color='crimson', width=2)))
                fig_r2.add_trace(go.Scatter(x=risk_df['year'], y=risk_df['injuries_per_100k_veh'],
                                            mode='lines+markers', name='Injuries',
                                            line=dict(color='orange', width=2)))
                fig_r2.update_layout(title='Per 100,000 Registered Vehicles',
                                     xaxis_title='Year', yaxis_title='Count',
                                     legend=dict(orientation='h'))
                st.plotly_chart(fig_r2, use_container_width=True)

            st.markdown("#### 🔮 2026 Normalized Risk Forecast")
            col_rf1, col_rf2 = st.columns(2)
            with col_rf1:
                lr = LinearRegression()
                lr.fit(risk_df[['year']], risk_df['deaths_per_100k_pop'])
                pred_risk_2026 = lr.predict([[2026]])[0]
                actual_risk_2025 = risk_df[risk_df['year'] == 2025]['deaths_per_100k_pop'].values[0]
                change_risk = (pred_risk_2026 - actual_risk_2025) / actual_risk_2025 * 100
                st.metric("Predicted Deaths per 100k Population (2026)",
                          f"{pred_risk_2026:.2f}", f"{change_risk:+.1f}% vs 2025")
            with col_rf2:
                lr2 = LinearRegression()
                lr2.fit(risk_df[['year']], risk_df['deaths_per_100k_veh'])
                pred_risk_veh_2026 = lr2.predict([[2026]])[0]
                actual_risk_veh_2025 = risk_df[risk_df['year'] == 2025]['deaths_per_100k_veh'].values[0]
                change_risk_veh = (pred_risk_veh_2026 - actual_risk_veh_2025) / actual_risk_veh_2025 * 100
                st.metric("Predicted Deaths per 100k Vehicles (2026)",
                          f"{pred_risk_veh_2026:.2f}", f"{change_risk_veh:+.1f}% vs 2025")

            st.info("💡 Insight: Total accidents may increase while normalized risk decreases — meaning roads are getting relatively safer as population and vehicle count grow.")
        else:
            st.warning("Normalized risk data could not be loaded.")

        st.divider()
        st.markdown("### 🗺️ Province-Level 2026–2030 Forecast")
        prov_pred_sel = st.selectbox("Select a province:", sorted(df_all['province'].unique(), key=tr_sort_key), key='pred_prov')
        prov_data_trend = df_all[df_all['province'] == prov_pred_sel].sort_values('year')

        if len(prov_data_trend) >= 3:
            X_prov = prov_data_trend['year'].values.reshape(-1, 1)
        
            # Accident forecast
            lr_prov = LinearRegression()
            lr_prov.fit(X_prov, prov_data_trend['total_accidents'].values)
            prov_forecasts = lr_prov.predict([[y] for y in forecast_years])
        
            # Death forecast
            lr_prov_deaths = LinearRegression()
            lr_prov_deaths.fit(X_prov, prov_data_trend['deaths'].values)
            prov_death_forecasts = lr_prov_deaths.predict([[y] for y in forecast_years])
            prov_death_forecasts = [max(0, int(round(v))) for v in prov_death_forecasts]

            actual_prov_2025 = prov_data_trend[prov_data_trend['year'] == 2025]['total_accidents'].values
            if len(actual_prov_2025) > 0:
                change_prov = (prov_forecasts[0] - actual_prov_2025[0]) / actual_prov_2025[0] * 100
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric(f"{prov_pred_sel} — 2026 Accident Forecast",
                              f"{int(round(prov_forecasts[0])):,}", f"{change_prov:+.1f}% vs 2025")
                with col_p2:
                    actual_deaths_2025 = prov_data_trend[prov_data_trend['year'] == 2025]['deaths'].values
                    if len(actual_deaths_2025) > 0:
                        change_deaths = (prov_death_forecasts[0] - actual_deaths_2025[0]) / actual_deaths_2025[0] * 100
                        st.metric(f"{prov_pred_sel} — 2026 Death Forecast",
                                  f"{prov_death_forecasts[0]:,}", f"{change_deaths:+.1f}% vs 2025")

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=prov_data_trend['year'], y=prov_data_trend['total_accidents'].astype(int),
                                  name='Actual', marker_color='steelblue'))
            fig3.add_trace(go.Bar(x=forecast_years, y=[int(round(v)) for v in prov_forecasts],
                                  name='Forecast', marker_color='red', opacity=0.7))
            fig3.update_layout(title=f'{prov_pred_sel} — Annual Accidents & 2026–2030 Forecast',
                               xaxis_title='Year', yaxis_title='Accidents', barmode='overlay',
                               yaxis=dict(tickformat=',d'))
            st.plotly_chart(fig3, use_container_width=True)

            # Province forecast table
            prov_forecast_table = pd.DataFrame({
                'Year': forecast_years,
                'Predicted Accidents': [f"{int(round(v)):,}" for v in prov_forecasts],
                'Predicted Deaths': [f"{v:,}" for v in prov_death_forecasts]
            })
            df_with_excel_download(prov_forecast_table, filename='province_forecast.csv', key='dl_tab5')

        model_desc = "Random Forest (year + registered vehicles + driver fault ratio)" if has_full_features else "Random Forest (year + registered vehicles)"
        st.info(f"⚙️ Model: {model_desc}")

    # ── TAB 6: Map ──────────────────────────────────────────────────
    case "Map":
        st.markdown('<div class="tab-desc">Visualise accident data across Turkey\'s 81 provinces on an interactive map.</div>', unsafe_allow_html=True)

        try:
            with open('turkey.geojson', encoding='utf-8') as f:
                turkey_geo = json.load(f)

            name_map = {
                'Afyonkarahisar': 'Afyon',
                'Mersin': 'İçel'
            }

            df_map = df_2025.copy()
            df_map['geo_name'] = df_map['province'].map(lambda x: name_map.get(x, x))

            col_m1, col_m2 = st.columns([1, 3])

            with col_m1:
                map_metric = st.radio("Select metric:",
                                      ["Total Accidents", "Deaths", "Deaths per 1000 Acc."],
                                      key='map_metric')
                map_col = {
                    'Total Accidents': 'total_accidents',
                    'Deaths': 'deaths',
                    'Deaths per 1000 Acc.': 'deaths_per_1000'
                }[map_metric]

                st.divider()
                st.markdown("**Top 5 Provinces**")
                top5 = df_2025.sort_values(map_col, ascending=False).head(5)
                for _, row in top5.iterrows():
                    val = f"{row[map_col]:,.2f}" if map_col == 'deaths_per_1000' else f"{row[map_col]:,.0f}"
                    st.markdown(f"<b>{row['province']}</b> — {val}", unsafe_allow_html=True)

            with col_m2:
                color_scales = {
                    'Total Accidents': 'Blues',
                    'Deaths': 'Reds',
                    'Deaths per 1000 Acc.': 'Oranges'
                }
                fig_map = px.choropleth(
                    df_map,
                    geojson=turkey_geo,
                    locations='geo_name',
                    featureidkey='properties.name',
                    color=map_col,
                    hover_name='province',
                    hover_data={
                        'total_accidents': ':,.0f',
                        'deaths': ':,.0f',
                        'deaths_per_1000': ':,.2f',
                        'geo_name': False
                    },
                    labels={
                        'total_accidents': 'Total Accidents',
                        'deaths': 'Deaths',
                        'deaths_per_1000': 'Deaths per 1,000 Acc.',
                        'geo_name': ''
                    },
                    color_continuous_scale=color_scales[map_metric],
                    title=f'Turkey — {map_metric} by Province (2025)'
                )
                fig_map.update_geos(
                    fitbounds="locations",
                    visible=True,
                    showland=True, landcolor='#0e1117',
                    showocean=True, oceancolor='#0e1117',
                    showlakes=False,
                    showcountries=True, countrycolor='#2d3548',
                    bgcolor='#0e1117'
                )
                fig_map.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=0),
                    coloraxis_colorbar=dict(title=map_metric),
                    dragmode='pan',
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#0e1117'
                )
                st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': False})
        except Exception as e:
            st.warning(f"Map could not be loaded: {e}")

    # ── TAB 7: Global Context ───────────────────────────────────────────────────
    case "Global Context":
        st.markdown('<div class="tab-desc">How does Turkey compare to the rest of the world in road safety? WHO data — always the latest available.</div>', unsafe_allow_html=True)
  
        who_df = load_who_data()
        latest_year = who_df['TimeDim'].max()
        who_latest = who_df[who_df['TimeDim'] == latest_year].copy()

        who_countries = who_latest[who_latest['SpatialDim'].str.len() == 3].copy()
        who_countries = who_countries[~who_countries['SpatialDim'].isin(['AFR','AMR','EMR','EUR','SEAR','WPR','GLOBAL'])]
        who_countries = who_countries.sort_values('NumericValue', ascending=False).reset_index(drop=True)
        who_countries['rank'] = who_countries.index + 1
        who_countries['country_name'] = who_countries['SpatialDim'].apply(get_country_name)

        tur_row = who_countries[who_countries['SpatialDim'] == 'TUR'].iloc[0]
        tur_rank = int(tur_row['rank'])
        tur_value = tur_row['NumericValue']
        world_avg = who_countries['NumericValue'].mean()
        pct_diff = (tur_value - world_avg) / world_avg * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Turkey's Rank", f"{tur_rank} / {len(who_countries)}")
        col2.metric("Turkey's Rate (per 100k)", f"{tur_value}")
        col3.metric("World Average", f"{world_avg:.1f}")

        if pct_diff < 0:
            st.success(f"🟢 Turkey's road traffic death rate is **{abs(pct_diff):.0f}% lower** than the world average — ranked safer than {len(who_countries) - tur_rank} out of {len(who_countries)-1} other countries.")
        else:
            st.warning(f"🔴 Turkey's road traffic death rate is **{pct_diff:.0f}% higher** than the world average.")

        fig_who = px.choropleth(who_countries, locations='SpatialDim', color='NumericValue',
                            color_continuous_scale='Reds',
                            title=f'Road Traffic Death Rate per 100k Population ({latest_year})',
                            hover_name='country_name',
                            hover_data={
                                'NumericValue': ':.1f',
                                'SpatialDim': False,
                                'rank': True
                            },
                            labels={
                                'NumericValue': 'Deaths per 100k',
                                'rank': 'World Rank'
                            })

        fig_who.add_trace(px.choropleth(
            who_countries[who_countries['SpatialDim'] == 'TUR'],
            locations='SpatialDim', color='NumericValue',
            color_continuous_scale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']]
        ).data[0])
        fig_who.data[-1].marker.line.color = 'blue'
        fig_who.data[-1].marker.line.width = 3
        fig_who.data[-1].showscale = False

        fig_who.update_geos(
            center=dict(lat=39, lon=35),
            projection_scale=2.5,
            visible=True,
            showland=True, landcolor='#1e2433',
            showocean=True, oceancolor='#0e1117',
            showlakes=False,
            showcountries=True, countrycolor='#2d3548',
            showframe=False,
            bgcolor='#0e1117'
        )

        fig_who.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            dragmode='pan',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117'
        )

        st.caption(f"📅 Data year: {latest_year} (WHO updates this periodically — page always shows the latest available)")
        st.plotly_chart(fig_who, use_container_width=True, config={'scrollZoom': False})
        st.caption("🔵 Turkey highlighted in blue")

        st.divider()
        st.markdown("### 🔍 Compare Turkey with Another Country")
        country_options = {f"{get_country_name(c)} ({c})": c for c in sorted(who_countries['SpatialDim'].unique())}
        selected_label = st.selectbox("Select a country to compare:", sorted(country_options.keys()))
        selected = country_options[selected_label]

        sel_row = who_countries[who_countries['SpatialDim'] == selected].iloc[0]
        comp_df = pd.DataFrame({
            'Country': ['Turkey', get_country_name(selected)],
            'Death Rate (per 100k)': [tur_value, sel_row['NumericValue']],
            'Rank': [tur_rank, int(sel_row['rank'])]
        })
        fig2_who = px.bar(comp_df, x='Country', y='Death Rate (per 100k)', color='Country',
                    title=f'Turkey vs {get_country_name(selected)}')
        st.plotly_chart(fig2_who, use_container_width=True)

        all_countries_df = who_countries[['rank', 'country_name', 'NumericValue']].copy()
        all_countries_df.columns = ['Rank', 'Country', 'Death Rate (per 100k)']
        df_with_excel_download(all_countries_df, filename='global_road_death_rates.csv', key='dl_tab7')

# ── Bottom table ─────────────────────────────────────────────────────────────

st.caption("Data: TÜİK — veriportali.tuik.gov.tr | Developer: Ertuğrul Halisdemir")

# ── Data Update Check ───────────────────────────────────────────────────────
st.divider()
tuik_latest = check_tuik_latest()
our_latest = int(df_all['year'].max())

if tuik_latest:
    if tuik_latest > our_latest:
        st.warning(f"🔔 New data available! TÜİK published the {tuik_latest} bulletin. Download `kaza_il_{tuik_latest}.xls` and related files from TÜİK and add them to the project folder.")
    else:
        st.success(f"✅ Data up to date — TÜİK latest bulletin: {tuik_latest} | Our dataset: {our_latest}")