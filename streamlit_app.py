import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime
import calendar
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

st.set_page_config(layout="wide", page_title="J&K Footfall Dashboard")

# Custom styling
st.markdown("""
<style>
    body {
        background-color: #f8f9fa;
    }
    .main {
        background-color: #ffffff;
    }
    h1 {
        color: #2c3e50 !important;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    h2, h3 {
        color: #34495e !important;
    }
    p, span, div {
        color: #2c3e50 !important;
    }
    /* Override metric styling for better visibility */
    [data-testid="metric-container"] {
        background-color: #f0f2f6 !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
    }
    [data-testid="metric-container"] > div > div > div {
        color: #2c3e50 !important;
    }
    [data-testid="metric-container"] > div {
        color: #2c3e50 !important;
    }
    /* Ensure all text is dark and visible */
    * {
        color: #2c3e50 !important;
    }
    /* Keep links blue */
    a {
        color: #3498db !important;
    }
    /* Plotly buttons and text */
    .modebar-btn {
        color: #2c3e50 !important;
    }
    .plotly-graph-div text {
        fill: #2c3e50 !important;
    }
    button, .button {
        color: #2c3e50 !important;
        background-color: #f0f2f6 !important;
    }
    /* Sidebar: force light background + dark text for all controls */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #2c3e50 !important;
    }
    section[data-testid="stSidebar"] .block-container,
    section[data-testid="stSidebar"] .css-1d391kg {
        background-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] .stButton>button,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] .stSlider {
        background-color: #f0f2f6 !important;
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎯 Jammu & Kashmir — Tourism Footfall Dashboard (2015–2025)")
EXCEL_PATH = os.path.join("data", "jk_tourism_website_extract.xlsx")  # path to your Excel
# Candidate column names the script will try to find:
DATE_COL_CANDIDATES = ["date", "Date", "month", "Month", "period"]
REGION_COL_CANDIDATES = ["region", "Region", "state", "State", "area", "Kashmir", "Jammu"]
DOMESTIC_COL_CANDIDATES = ["domestic", "Domestic", "domestic_count", "domestic_visitors", "domestic_tourists", "Domestic_Tourists"]
FOREIGN_COL_CANDIDATES = ["foreign", "Foreign", "foreign_count", "foreign_visitors", "foreign_tourists", "Foreign_Tourists"]
TOTAL_COL_CANDIDATES = ["total", "Total", "visitors", "count", "total_visitors", "total_tourists", "Total Tourists in J&K"]
# -----------------------------------------------------------------------------------

@st.cache
def load_excel(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}. Put your Excel at {path}")
    
    # Read all sheets and combine them
    xls = pd.ExcelFile(path)
    dfs = []
    
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        dfs.append(df)
    
    # Concatenate all sheets
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def prepare_df(raw):
    df = raw.copy()
    
    # Debug: show what columns were found
    print(f"DEBUG: Available columns: {df.columns.tolist()}")

    # Check if we have Year and Month columns (older sheets format)
    if "Year" in df.columns and "Month" in df.columns:
        # Get all regional columns first (those with Jammu or Kashmir in the name)
        regional_cols = [col for col in df.columns if ('jammu' in col.lower() or 'kashmir' in col.lower()) and col.lower() != 'total tourists in j&k']
        
        if regional_cols:
            print(f"DEBUG: Found {len(regional_cols)} regional columns: {regional_cols}")
            
            # Convert month names to numbers
            def convert_month(m):
                if pd.isna(m):
                    return None
                if isinstance(m, str):
                    try:
                        return list(calendar.month_name).index(m)
                    except ValueError:
                        return None
                # Already numeric
                try:
                    return int(m)
                except:
                    return None
            
            # Convert Year to int, handling NaN values
            df["year"] = pd.to_numeric(df["Year"], errors="coerce")
            # Convert Month
            df["month_num"] = df["Month"].apply(convert_month)
            
            print(f"DEBUG: After conversion - year dtype: {df['year'].dtype}, month_num dtype: {df['month_num'].dtype}")
            print(f"DEBUG: year nulls: {df['year'].isna().sum()}, month_num nulls: {df['month_num'].isna().sum()}")
            print(f"DEBUG: Sample years: {df['year'].head()}")
            print(f"DEBUG: Sample months: {df['month_num'].head()}")
            
            # Filter to valid year/month combinations
            df = df.dropna(subset=['year', 'month_num']).copy()
            print(f"DEBUG: df shape after dropna: {df.shape}")
            
            # Create date column
            df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-" + df['month_num'].astype(int).astype(str) + "-01", errors="coerce")
            df["month"] = df["month_num"].astype(int)
            
            # Drop any remaining rows with invalid dates
            df = df.dropna(subset=['date']).copy()
            print(f"DEBUG: df shape after date creation: {df.shape}")
            
            if df.empty:
                print("DEBUG: No valid data after year/month/date conversion!")
                return pd.DataFrame()
            
            # Unpivot regional columns to long format
            id_cols = ["date", "year", "month"]
            
            df_melted = df.melt(id_vars=id_cols, value_vars=regional_cols, var_name='region_metric', value_name='count')
            print(f"DEBUG: df_melted shape after melt: {df_melted.shape}")
            print(f"DEBUG: df_melted sample:\n{df_melted.head()}")
            
            # Parse region and metric from column names - handle case insensitivity
            def parse_region_metric(col_name):
                col_lower = col_name.lower()
                # Determine region (Jammu or Kashmir)
                if col_lower.startswith('jammu'):
                    region = 'Jammu'
                    metric = col_lower.replace('jammu_', '').replace('jammu', '')
                elif col_lower.startswith('kashmir'):
                    region = 'Kashmir'
                    metric = col_lower.replace('kashmir_', '').replace('kashmir', '')
                else:
                    return None, None
                
                # Clean up metric
                metric = metric.strip('_').lower()
                if metric.startswith('_'):
                    metric = metric[1:]
                
                return region, metric
            
            # Apply parsing and create new columns
            parsed_data = df_melted['region_metric'].apply(lambda x: parse_region_metric(x))
            df_melted['region'] = parsed_data.apply(lambda x: x[0] if x else None)
            df_melted['metric'] = parsed_data.apply(lambda x: x[1] if x else None)
            
            print(f"DEBUG: Before filtering - {len(df_melted)} rows")
            print(f"DEBUG: Sample region_metric values: {df_melted['region_metric'].head(10).tolist()}")
            print(f"DEBUG: Sample parsed region values: {df_melted['region'].head(10).tolist()}")
            print(f"DEBUG: Sample parsed metric values: {df_melted['metric'].head(10).tolist()}")
            
            # Filter out rows where parsing failed
            df_melted = df_melted.dropna(subset=['region', 'metric']).copy()
            df_melted['count'] = pd.to_numeric(df_melted['count'], errors='coerce').fillna(0)
            
            print(f"DEBUG: df_melted after parsing has {len(df_melted)} rows")
            if len(df_melted) > 0:
                print(f"DEBUG: Unique metrics: {df_melted['metric'].unique().tolist()}")
            
            # Create separate columns for domestic, foreign, total
            result_rows = []
            for _, row in df_melted.iterrows():
                result_rows.append({
                    'date': row['date'],
                    'year': row['year'],
                    'month': row['month'],
                    'region': row['region'],
                    'metric': row['metric'].lower() if isinstance(row['metric'], str) else '',
                    'value': row['count']
                })
            
            df_unpivot = pd.DataFrame(result_rows)
            print(f"DEBUG: df_unpivot has {len(df_unpivot)} rows")
            if len(df_unpivot) == 0:
                print("DEBUG: df_unpivot is empty, cannot pivot")
                return pd.DataFrame()
            
            df_final = df_unpivot.pivot_table(
                index=['date', 'year', 'month', 'region'],
                columns='metric',
                values='value',
                aggfunc='sum'
            ).reset_index()
            
            # Ensure all columns exist
            for col in ['domestic', 'foreign', 'total']:
                if col not in df_final.columns:
                    df_final[col] = 0
            
            # Handle total if not calculated
            if (df_final['total'] == 0).all():
                df_final['total'] = df_final['domestic'] + df_final['foreign']
            
            print(f"DEBUG: Processed {len(df_final)} rows from regional columns")
            print(f"DEBUG: Year range: {df_final['year'].min()} - {df_final['year'].max()}")
            print(f"DEBUG: Regions: {df_final['region'].unique().tolist()}")
            
            if not df_final.empty:
                return df_final[["date","year","month","region","domestic","foreign","total"]]
            else:
                print("DEBUG: df_final is empty after processing regional columns")
    
    # Fallback to original logic
    date_col = pick_col(df, DATE_COL_CANDIDATES)
    region_col = pick_col(df, REGION_COL_CANDIDATES)
    domestic_col = pick_col(df, DOMESTIC_COL_CANDIDATES)
    foreign_col = pick_col(df, FOREIGN_COL_CANDIDATES)
    total_col = pick_col(df, TOTAL_COL_CANDIDATES)

    print(f"DEBUG: Detected - date_col: {date_col}, region_col: {region_col}, total_col: {total_col}")

    if date_col is None:
        st.error("Couldn't find a date column. Edit DATE_COL_CANDIDATES in the script to match your Excel.")
        st.error(f"Available columns: {df.columns.tolist()}")
        st.stop()

    # normalize date column to month start
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # if year/month present but dates are missing, try to construct:
    if df[date_col].isna().all():
        # try year & month columns
        if "Year" in df.columns or "year" in df.columns:
            ycol = "Year" if "Year" in df.columns else "year"
            if "Month" in df.columns or "month" in df.columns:
                mcol = "Month" if "Month" in df.columns else "month"
                # Handle month names (January, February, etc.)
                def convert_month(m):
                    if isinstance(m, str):
                        try:
                            return list(calendar.month_name).index(m)
                        except ValueError:
                            return None
                    return int(m) if pd.notna(m) else None
                
                df['month_num'] = df[mcol].apply(convert_month)
                df[date_col] = pd.to_datetime(df[ycol].astype(str) + "-" + df['month_num'].astype(str) + "-01", errors="coerce")
                df = df.drop('month_num', axis=1)

    df = df.dropna(subset=[date_col])
    df["date"] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Region
    if region_col is None:
        # fallback: if there's no region column, create "J&K" as single region
        df["region"] = "J&K"
    else:
        df["region"] = df[region_col].astype(str)

    # Total/domestic/foreign
    if total_col is None:
        # try compute from domestic/foreign
        if domestic_col and foreign_col:
            df["domestic"] = pd.to_numeric(df[domestic_col], errors="coerce").fillna(0)
            df["foreign"] = pd.to_numeric(df[foreign_col], errors="coerce").fillna(0)
            df["total"] = df["domestic"] + df["foreign"]
        else:
            # if only one count column present, use it as total
            any_count = domestic_col or foreign_col
            if any_count:
                df["total"] = pd.to_numeric(df[any_count], errors="coerce").fillna(0)
                df["domestic"] = 0
                df["foreign"] = 0
            else:
                df["total"] = 0
                df["domestic"] = 0
                df["foreign"] = 0
    else:
        df["total"] = pd.to_numeric(df[total_col], errors="coerce").fillna(0)
        # if domestic/foreign present, keep them
        if domestic_col:
            df["domestic"] = pd.to_numeric(df[domestic_col], errors="coerce").fillna(0)
        else:
            df["domestic"] = 0
        if foreign_col:
            df["foreign"] = pd.to_numeric(df[foreign_col], errors="coerce").fillna(0)
        else:
            df["foreign"] = 0

    return df[["date","year","month","region","domestic","foreign","total"]]

# Load and prepare
raw_df = load_excel(EXCEL_PATH)
df = prepare_df(raw_df)

# Check if data is empty
if df.empty:
    st.error("❌ No data found. Please check your Excel file and column names.")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("📊 Filters")
    years = sorted(df["year"].dropna().unique().astype(int))
    if len(years) == 0:
        st.error("No valid year data found. Please check your Excel file structure.")
        st.stop()
    min_year, max_year = int(min(years)), int(max(years))
    start_year, end_year = st.slider("Year range", min_value=min_year, max_value=max_year,
                                     value=(min_year, max_year))
    regions = st.multiselect("Regions", options=sorted(df["region"].unique()), default=sorted(df["region"].unique()))
    show_type = st.selectbox("Visitor Type", ["Total", "Domestic", "Foreign"], index=0)
    month_filter = st.selectbox("Filter by Month", ["All"] + list(calendar.month_name[1:]), index=0)
    # heatmap colorscale selection (avoid red as 'good')
    heatmap_colorscale = st.selectbox(
        "Heatmap colorscale (avoid red = 'good')",
        options=["Viridis", "YlGnBu", "Cividis", "Plasma", "Blues"],
        index=0,
        help="Pick a colorscale where red isn't the primary 'good' color. Default = Viridis"
    )
    forecast_years = st.slider("Forecast Years Ahead", 1, 5, 5)
    # Debug toggles
    debug_heatmap = st.checkbox("Show heatmap debug info", value=False)

# Filter
mask = (df["year"] >= start_year) & (df["year"] <= end_year) & (df["region"].isin(regions))
if month_filter != "All":
    mnum = list(calendar.month_name).index(month_filter)
    mask &= (df["month"] == mnum)
df_f = df.loc[mask].copy()

# Metric
metric_col = {"Total":"total","Domestic":"domestic","Foreign":"foreign"}[show_type]

# KPI row
st.markdown("### 📈 Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)

total_val = int(df_f[metric_col].sum())
k1.metric("👥 Total Visitors", f"{total_val:,}")

avg_monthly = int(df_f[metric_col].mean()) if not df_f.empty else 0
k2.metric("📅 Average Monthly", f"{avg_monthly:,}")

yoy = None
if not df_f.empty:
    last_year = df_f["year"].max()
    prev_year = last_year - 1
    s_last = df_f[df_f["year"]==last_year][metric_col].sum()
    s_prev = df_f[df_f["year"]==prev_year][metric_col].sum()
    if s_prev > 0:
        yoy = (s_last - s_prev) / s_prev * 100
        yoy_delta = f"{yoy:+.2f}%"
    else:
        yoy_delta = "N/A"
else:
    yoy_delta = "N/A"

k3.metric("📊 YoY Growth", yoy_delta)

peak_val = "N/A"
if not df_f.empty:
    monthly = df_f.groupby(["year","month"])[metric_col].sum().reset_index()
    if len(monthly) > 0:
        idx_max = monthly[metric_col].idxmax()
        peak = monthly.loc[idx_max]
        peak_val = f"{int(peak[metric_col]):,}"
        
k4.metric("🏆 Peak Month", peak_val)

# Time series (bar + line overlay)
ts = df_f.groupby("date")[metric_col].sum().reset_index()
if not ts.empty:
    fig_ts = go.Figure()
    # Bars for monthly values (slightly transparent)
    fig_ts.add_trace(go.Bar(
        x=ts["date"],
        y=ts[metric_col],
        name="Monthly",
        marker_color='rgba(31,119,180,0.35)',
        hovertemplate='%{y:,}<extra></extra>'
    ))
    # Line overlay for trend
    fig_ts.add_trace(go.Scatter(
        x=ts["date"],
        y=ts[metric_col],
        mode='lines+markers',
        name=f"{show_type} Trend",
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6, color='#1f77b4')
    ))

    fig_ts.update_layout(
        barmode='overlay',
        title={
            'text': f"{show_type} Visitors Over Time",
            'x': 0.01,
            'xanchor': 'left',
            'font': {'color': '#2c3e50', 'size': 16}
        },
        xaxis_title="Date",
        yaxis_title="Number of Visitors",
        hovermode='x unified',
        template='plotly_white',
        font=dict(size=12, color='#2c3e50'),
        height=420,
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        margin=dict(l=50, r=50, t=100, b=50)
    )
    fig_ts.update_xaxes(title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50'))
    fig_ts.update_yaxes(title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50'))
    st.plotly_chart(fig_ts, use_container_width=True)

# Region comparison
comp = df_f.groupby(["date","region"])[metric_col].sum().reset_index()
if not comp.empty:
    color_map = {'Jammu': '#e74c3c', 'Kashmir': '#3498db'}
    # pivot for bars per region
    comp_pivot = comp.pivot(index='date', columns='region', values=metric_col).fillna(0)
    fig_comp = go.Figure()
    for region in comp_pivot.columns:
        fig_comp.add_trace(go.Bar(
            x=comp_pivot.index,
            y=comp_pivot[region],
            name=region,
            marker_color=color_map.get(region, None),
            opacity=0.7,
        ))

    # Add a thin total line for emphasis
    total_series = comp.groupby('date')[metric_col].sum().reset_index()
    fig_comp.add_trace(go.Scatter(
        x=total_series['date'],
        y=total_series[metric_col],
        mode='lines',
        name='Total',
        line=dict(color='#2c3e50', width=2, dash='dash')
    ))

    fig_comp.update_layout(
        barmode='group',
        title={
            'text': f"Region-wise {show_type} Visitor Trends",
            'x': 0.01,
            'xanchor': 'left',
            'font': {'color': '#2c3e50', 'size': 16}
        },
        xaxis_title='Date',
        yaxis_title='Number of Visitors',
        hovermode='x unified',
        template='plotly_white',
        font=dict(size=12, color='#2c3e50'),
        height=420,
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        margin=dict(l=50, r=50, t=100, b=50)
    )
    fig_comp.update_xaxes(title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50'))
    fig_comp.update_yaxes(title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50'))
    st.plotly_chart(fig_comp, use_container_width=True)

# Seasonal heatmap
pivot = df_f.pivot_table(index="year", columns="month", values=metric_col, aggfunc="sum", fill_value=0)
if not pivot.empty:
    # Ensure months 1..12 exist as columns (some sheets may miss months)
    pivot = pivot.sort_index()
    full_months = list(range(1, 13))
    pivot = pivot.reindex(columns=full_months, fill_value=0)

    # Server-side debug: always write pivot to a CSV for diagnostics
    try:
        debug_path = os.path.join("data", "heatmap_pivot_debug.csv")
        pivot.to_csv(debug_path)
        print(f"DEBUG: Wrote heatmap pivot to {debug_path}")
    except Exception as e:
        print(f"DEBUG: Failed to write pivot debug file: {e}")

    # Print a concise pivot summary to server stdout to help debugging
    try:
        print(f"DEBUG: pivot.shape: {pivot.shape}, pivot.sum: {pivot.values.sum()}")
    except Exception as e:
        print(f"DEBUG: Failed writing pivot summary: {e}")

    # If there's no non-zero data, show an informative message
    if pivot.values.sum() == 0:
        st.info("No visitor data available for the selected filters to show in the heatmap.")
        if 'debug_heatmap' in locals() and debug_heatmap:
            st.write("pivot.shape:", pivot.shape)
            st.write("pivot.columns:", pivot.columns.tolist())
            st.write("pivot.index:", pivot.index.astype(str).tolist())
            st.write("pivot.sum:", pivot.values.sum())
    else:
        # Convert month numbers to abbreviated names
        month_labels = [calendar.month_abbr[i] for i in full_months]
        years = pivot.index.astype(int).tolist()

        # Try using px.imshow first; fall back to go.Heatmap if there's any issue
        # Render heatmap using go.Heatmap with explicit z-range and colorbar formatting
        try:
            zmin = 0
            zmax = float(pivot.values.max()) if pivot.values.size else 0
            hm = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=month_labels,
                y=years,
                colorscale=heatmap_colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(tickfont=dict(color='#2c3e50'), title='Visitors', tickformat=','),
                hovertemplate='%{y} - %{x}: %{z:,}<extra></extra>'
            ))
            hm.update_layout(
                title='Seasonal Heatmap: Visitor Distribution',
                font=dict(size=12, color='#2c3e50'),
                height=420,
                xaxis_title='Month',
                yaxis_title='Year',
                paper_bgcolor='rgba(240,240,240,0.3)',
                plot_bgcolor='rgba(255,255,255,1)'
            )
            hm.update_xaxes(side='bottom', title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50'))
            hm.update_yaxes(title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50'))
            st.plotly_chart(hm, use_container_width=True)
        except Exception as e:
            st.write("Heatmap rendering failed:", e)

        # Server-rendered static PNG fallback (matplotlib) - useful when plotly client rendering fails
        try:
            static_path = os.path.join("data", "heatmap_static.png")
            # Map Plotly colorscale names to matplotlib cmaps for the static fallback
            mpl_cmap_map = {
                "Viridis": "viridis",
                "YlGnBu": "YlGnBu",
                "Cividis": "cividis",
                "Plasma": "plasma",
                "Blues": "Blues"
            }
            cmap_name = mpl_cmap_map.get(heatmap_colorscale, "viridis")

            # Create a matplotlib heatmap to guarantee a visual on the server
            fig, ax = plt.subplots(figsize=(10, max(3, len(years) * 0.35)))
            im = ax.imshow(pivot.values, aspect='auto', cmap=cmap_name, origin='lower')

            # X ticks -> months
            ax.set_xticks(np.arange(len(month_labels)))
            ax.set_xticklabels(month_labels, rotation=45, ha='right')

            # Y ticks -> years
            ax.set_yticks(np.arange(len(years)))
            ax.set_yticklabels(years)

            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            ax.set_title('Seasonal Heatmap (static PNG fallback)')

            # Colorbar with thousands separator
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

            fig.tight_layout()
            fig.savefig(static_path, dpi=150)
            plt.close(fig)
            st.image(static_path, use_column_width=True)
        except Exception as e:
            print(f"DEBUG: Matplotlib heatmap failed: {e}")

# ---------- Forecasting: train ONLY on 2023,2024,2025 and forecast next N years (controlled by sidebar slider) ----------
TRAIN_YEARS = [2023, 2024, 2025]

def make_prophet_forecast(df_source, region_name, metric_col, years_ahead=5, cap_multiplier=1.2, fallback_cap=1_000_000):
    """
    Train Prophet using logistic growth (plateauing) safely.

    Parameters:
      - df_source: original dataframe (must contain 'date','year','region', metric_col)
      - region_name: region to train on (e.g., 'Kashmir' or 'Jammu')
      - metric_col: column name to forecast ('total' / 'domestic' / 'foreign')
      - years_ahead: integer horizon in years
      - cap_multiplier: multiplier for cap relative to observed max (default 1.2)
      - fallback_cap: fallback CAP if historical max is missing or invalid

    Returns: (model, forecast_df, last_train_date) or (None, None, None) if no training data
    """
    # 1) Filter training data to the fixed years (2023-2025)
    train_df = df_source[(df_source["year"].isin(TRAIN_YEARS)) & (df_source["region"] == region_name)].copy()
    if train_df.empty:
        return None, None, None

    # 2) Aggregate to regular monthly series (month-start)
    monthly = train_df.groupby("date")[metric_col].sum().reset_index()
    monthly = monthly.set_index("date").resample("MS").sum().reset_index()
    monthly = monthly.rename(columns={metric_col: "y", "date": "ds"})

    # Ensure y is numeric and non-negative
    monthly["y"] = pd.to_numeric(monthly["y"], errors="coerce").fillna(0)
    monthly["y"] = monthly["y"].clip(lower=0)

    # 3) Compute a safe CAP (capacity) for logistic growth
    max_obs = monthly["y"].max() if "y" in monthly.columns else None
    if pd.isna(max_obs) or max_obs <= 0:
        CAPACITY = float(fallback_cap)
    else:
        CAPACITY = float(max_obs) * float(cap_multiplier)
        # guard: ensure CAPACITY > max_obs
        if CAPACITY <= max_obs:
            CAPACITY = float(max_obs) * 1.05

    # Cap must be a float and > 0 for Prophet logistic growth
    if CAPACITY <= 0:
        CAPACITY = float(fallback_cap)

    monthly["cap"] = float(CAPACITY)

    # 4) Build and fit Prophet with logistic growth and additive seasonality
    m = Prophet(
        growth="logistic",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive"
    )
    # monthly-like seasonality to capture intra-year variations (keeps it stable)
    m.add_seasonality(name="monthly_effect", period=30.5, fourier_order=5)

    # Fit requires ds, y, cap
    try:
        m.fit(monthly[["ds", "y", "cap"]])
    except Exception as e:
        # fitting can fail if data is degenerate; return None so calling code can handle
        print(f"Prophet fit error for region {region_name}: {e}")
        return None, None, None

    # 5) Create future frame and ensure cap column is present for all rows
    future = m.make_future_dataframe(periods=years_ahead * 12, freq="MS")
    future["cap"] = float(CAPACITY)

    # 6) Predict and post-process to keep forecasts realistic
    forecast = m.predict(future)

    # Clip forecasts to [0, CAPACITY] to avoid negative or >cap values because of numerical reasons
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in forecast.columns:
            forecast[col] = forecast[col].clip(lower=0, upper=CAPACITY)

    last_train_date = monthly["ds"].max() if not monthly["ds"].empty else None

    return m, forecast, last_train_date


st.markdown("## 🔮 Visitor Forecast (trained only on 2023–2025)")
st.markdown("Forecasts are produced per-region using only months from 2023, 2024 and 2025 as training data.")

avail_regions = sorted(df["region"].unique())
regions_to_forecast = st.multiselect("Regions to forecast (training limited to 2023-2025)", options=avail_regions, default=avail_regions)

if len(regions_to_forecast) == 0:
    st.info("Select at least one region to generate forecasts.")
else:
    cols = st.columns(len(regions_to_forecast))
    summary_rows = []
    for i, reg in enumerate(regions_to_forecast):
        with cols[i]:
            st.markdown(f"### ➤ {reg} forecast")
            model, fcst, last_train = make_prophet_forecast(df, reg, metric_col, years_ahead=forecast_years)
            if model is None:
                st.warning(f"No training rows for {reg} in years {TRAIN_YEARS}. Can't forecast.")
                continue

            # Interactive Prophet plot
            try:
                fig_prophet = plot_plotly(model, fcst)
                fig_prophet.update_layout(
                    title=f"{reg} — Monthly forecast (trained on {TRAIN_YEARS})",
                    height=420,
                    template="plotly_white",
                    margin=dict(t=80)
                )
                st.plotly_chart(fig_prophet, use_container_width=True)
            except Exception as e:
                st.write("Interactive forecast plot failed:", e)

            # Compact yearly forecast table for the next N years
            fcst_monthly = fcst.copy()
            fcst_monthly["year"] = fcst_monthly["ds"].dt.year
            if last_train is not None:
                forecast_start = (last_train + pd.DateOffset(months=1)).replace(day=1)
            else:
                forecast_start = fcst_monthly["ds"].min()

            future_mask = fcst_monthly["ds"] >= forecast_start
            future_monthly = fcst_monthly.loc[future_mask, ["ds", "yhat"]].copy()
            if future_monthly.empty:
                st.info("No future months in forecast (check training data).")
                continue

            future_monthly["year"] = future_monthly["ds"].dt.year
            yearly_forecast = future_monthly.groupby("year")["yhat"].sum().reset_index()
            last_year_in_train = last_train.year if last_train is not None else df["year"].max()
            target_years = list(range(last_year_in_train + 1, last_year_in_train + 1 + forecast_years))
            yearly_forecast = yearly_forecast[yearly_forecast["year"].isin(target_years)].copy()
            yearly_forecast["yhat"] = yearly_forecast["yhat"].round().astype(int)

            st.table(yearly_forecast.rename(columns={"year": "Year", "yhat": "Predicted visitors (sum)"}).set_index("Year"))

            # --- Download button for this region's yearly forecast (CSV) ---
            try:
                csv_region = yearly_forecast.rename(columns={"year": "Year", "yhat": "Predicted visitors (sum)"}).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {reg} yearly forecast CSV",
                    data=csv_region,
                    file_name=f"{reg.lower().replace(' ', '_')}_yearly_forecast.csv",
                    mime='text/csv'
                )
            except Exception as e:
                st.write("Download button for regional forecast failed:", e)

            for _, r in yearly_forecast.iterrows():
                summary_rows.append({"region": reg, "year": int(r["year"]), "predicted": int(r["yhat"])})

    # Combined summary table across regions
    if summary_rows:
        st.markdown("### 📋 Combined Forecast Summary")
        summary_df = pd.DataFrame(summary_rows)
        pivot_sum = summary_df.pivot_table(index="year", columns="region", values="predicted", aggfunc="sum", fill_value=0).reset_index().rename(columns={"year":"Year"})
        st.dataframe(pivot_sum.style.format("{:,}"), height=300)
        
        # --- Download button for combined forecast summary (CSV) ---
        try:
            csv_combined = pivot_sum.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download combined forecast CSV",
                data=csv_combined,
                file_name="jk_combined_forecast_summary.csv",
                mime='text/csv'
            )
        except Exception as e:
            st.write("Download button for combined forecast failed:", e)
# ------------------------------------------------------------------------------------------
