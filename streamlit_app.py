# streamlit_app.py
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
    forecast_years = st.slider("Forecast Years Ahead", 1, 5, 2)
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
                colorscale='YlOrRd',
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
            # Create a matplotlib heatmap to guarantee a visual on the server
            fig, ax = plt.subplots(figsize=(10, max(3, len(years) * 0.35)))
            im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', origin='lower')

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

            plt.tight_layout()
            fig.savefig(static_path, dpi=150)
            plt.close(fig)

            # Display static image below the interactive Plotly heatmap (fallback)
            try:
                st.image(static_path, use_column_width=True, caption="Static PNG fallback heatmap (server-rendered)")
            except Exception as e:
                print(f"DEBUG: Failed to display static PNG in Streamlit: {e}")
        except Exception as e:
            print(f"DEBUG: Failed to create static heatmap PNG: {e}")

        # If debug flag is on, also show the pivot table and a go.Heatmap for verification
        if 'debug_heatmap' in locals() and debug_heatmap:
            st.write("pivot (sample):")
            st.dataframe(pivot.head())
            try:
                hm_debug = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=month_labels,
                    y=years,
                    colorscale='YlOrRd'
                ))
                hm_debug.update_layout(title='Heatmap (debug go.Heatmap)')
                st.plotly_chart(hm_debug, use_container_width=True)
            except Exception as e:
                st.write("Failed to render debug go.Heatmap:", e)

# Peak & lean by region
st.markdown("### 🎯 Peak & Lean Months by Region")
for region in regions:
    df_r = df_f[df_f["region"]==region]
    if df_r.empty:
        st.info(f"No data available for **{region}** in selected filters")
        continue
    
    monthly = df_r.groupby(["year","month"])[metric_col].sum().reset_index()
    if len(monthly) == 0:
        continue
    
    idx_max = monthly[metric_col].idxmax()
    idx_min = monthly[metric_col].idxmin()
    peak = monthly.loc[idx_max]
    lean = monthly.loc[idx_min]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            f"🏆 {region} — Peak",
            f"{int(peak[metric_col]):,}",
            f"{int(peak['year'])}-{int(peak['month'])}"
        )
    with col2:
        st.metric(
            f"📉 {region} — Lean",
            f"{int(lean[metric_col]):,}",
            f"{int(lean['year'])}-{int(lean['month'])}"
        )

# Forecasting (Prophet)
st.markdown("---")
st.markdown("## 🔮 Visitor Forecast (Next Years)")
if df_f.empty:
    st.info("No data selected for forecasting. Adjust filters to see predictions.")
else:
    with st.spinner("🤖 Training Prophet forecast model..."):
        ts_forecast_df = df_f.groupby("date")[metric_col].sum().reset_index().rename(columns={"date":"ds", metric_col:"y"})
        ts_forecast_df = ts_forecast_df.sort_values("ds")
        full_idx = pd.date_range(start=ts_forecast_df["ds"].min(), end=ts_forecast_df["ds"].max(), freq="MS")
        ts_forecast_df = ts_forecast_df.set_index("ds").reindex(full_idx).rename_axis("ds").fillna(0).reset_index()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(ts_forecast_df.rename(columns={"ds":"ds","y":"y"}))
        future = m.make_future_dataframe(periods=12*forecast_years, freq="MS")
        forecast = m.predict(future)
        
        fig_f = plot_plotly(m, forecast)
        fig_f.update_layout(
            title="Visitor Forecast with Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Forecasted Visitors",
            template='plotly_white',
            font=dict(size=12, color='#2c3e50'),
            height=500,
            hovermode='x unified',
            paper_bgcolor='rgba(240,240,240,0.3)',
            plot_bgcolor='rgba(255,255,255,1)',
        )
        # Add observed historical bars behind the forecast for readability
        try:
            fig_f.add_trace(go.Bar(
                x=ts_forecast_df['ds'],
                y=ts_forecast_df['y'],
                name='Observed',
                marker_color='rgba(31,119,180,0.35)',
                opacity=0.6,
                hovertemplate='%{y:,}<extra></extra>'
            ))
            # overlay bars behind lines
            fig_f.update_layout(barmode='overlay')
        except Exception:
            # if anything goes wrong skip adding bars (keep forecast plot functional)
            pass
        # Ensure all text elements are dark colored
        fig_f.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50'))
        fig_f.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', title_font=dict(color='#2c3e50'), tickfont=dict(color='#2c3e50'))
        st.plotly_chart(fig_f, use_container_width=True)
        
        # Get future forecast (beyond training data)
        max_date = ts_forecast_df["ds"].max()
        future_forecast = forecast[forecast["ds"] > max_date].copy()
        if not future_forecast.empty:
            future_totals = future_forecast.set_index("ds")["yhat"].resample("Y").sum()
            st.markdown("### 📊 Predicted Annual Totals")
            
            forecast_table = pd.DataFrame({
                "Year": [d.year for d in future_totals.index],
                "Forecasted Visitors": future_totals.round().astype(int).values
            })
            
            st.dataframe(forecast_table)

st.markdown("---")
st.markdown("### 📥 Export Data")
csv_data = df_f.to_csv(index=False)
st.download_button(
    label="⬇️ Download Filtered Data (CSV)",
    data=csv_data.encode("utf-8"),
    file_name=f"jk_footfall_{start_year}_{end_year}.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>
    <p>Jammu & Kashmir Tourism Footfall Dashboard | Data Period: 2015-2025</p>
    <p>Last Updated: December 2025</p>
</div>
""", unsafe_allow_html=True)
