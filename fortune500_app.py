import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fortune 500 Analytics Dashboard",
    page_icon="WhatsApp Image 2026-02-11 at 3.17.53 PM.jpeg",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_path = r"WhatsApp Image 2026-02-11 at 3.32.24 PM.jpeg"
image_base64 = get_base64_of_image(image_path)

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{image_base64}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}}

.main > div {{
    background: rgba(0, 0, 0, 0.65) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    margin: 10px !important;
}}

.css-1d391kg, .css-163ttbj, [data-testid="stSidebar"] > div:first-child {{
    background: rgba(10, 10, 20, 0.85) !important;
    backdrop-filter: blur(10px) !important;
    border-right: 1px solid rgba(255,255,255,0.15) !important;
}}

.custom-card {{
    background: rgba(20, 25, 40, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}}

.custom-card h1, .custom-card h2, .custom-card h3, .custom-card h4, 
.custom-card h5, .custom-card h6, .custom-card p, .custom-card span, 
.custom-card div {{
    color: #ffffff !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.3) !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    transition: all 0.3s ease !important;
}}

.stButton > button:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 20px rgba(74, 85, 104, 0.4) !important;
    background: linear-gradient(135deg, #2D3748 0%, #1A202C 100%) !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 12px;
    background: rgba(0,0,0,0.3);
    padding: 8px;
    border-radius: 16px;
    backdrop-filter: blur(5px);
}}

.stTabs [data-baseweb="tab"] {{
    background: rgba(255,255,255,0.1);
    border-radius: 12px;
    color: white !important;
    padding: 12px 24px;
    border: 1px solid rgba(255,255,255,0.15);
    font-weight: 500;
    transition: all 0.3s ease;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%) !important;
    color: white !important;
    border: none;
    box-shadow: 0 4px 12px rgba(74, 85, 104, 0.3);
}}

.stSelectbox, .stDropdown {{
    background: rgba(30, 35, 50, 0.8);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(5px);
}}

.stSelectbox label, .stDropdown label {{
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
}}

.stSelectbox > div > div {{
    background: rgba(40, 45, 60, 0.9) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
}}

h1, h2, h3, h4, h5, h6 {{
    color: #ffffff !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
    letter-spacing: 0.5px !important;
}}

.stMarkdown {{
    color: #ffffff !important;
}}

.stMarkdown p, .stMarkdown span {{
    color: rgba(255,255,255,0.95) !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
}}

.stMetric {{
    background: rgba(30, 35, 50, 0.7) !important;
    backdrop-filter: blur(8px) !important;
    padding: 20px !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
}}

.stMetric label {{
    color: rgba(255,255,255,0.9) !important;
    font-size: 0.9rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}}

.stMetric div {{
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.3) !important;
}}

.dataframe, .stDataFrame {{
    background: rgba(30, 35, 50, 0.8) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: 12px !important;
    padding: 10px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}}

.stDataFrame td, .stDataFrame th {{
    color: #ffffff !important;
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    padding: 12px !important;
}}

.stDataFrame th {{
    background: rgba(74, 85, 104, 0.3) !important;
    color: white !important;
    font-weight: 600 !important;
}}

.stSuccess, .stInfo {{
    background: rgba(30, 35, 50, 0.8) !important;
    backdrop-filter: blur(8px) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
}}

.footer {{
    text-align: center;
    color: rgba(255,255,255,0.9) !important;
    padding: 25px;
    font-size: 14px;
    background: rgba(0,0,0,0.5);
    backdrop-filter: blur(5px);
    border-radius: 15px;
    margin-top: 30px;
}}

.developer {{
    background: linear-gradient(135deg, rgba(160, 174, 192, 0.2) 0%, rgba(113, 128, 150, 0.2) 100%) !important;
    backdrop-filter: blur(10px) !important;
    padding: 20px !important;
    border-radius: 16px !important;
    margin-top: 25px !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
}}

.stRadio > div {{
    background: rgba(30, 35, 50, 0.6) !important;
    backdrop-filter: blur(8px) !important;
    padding: 15px !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}}

.stRadio label {{
    color: white !important;
    font-size: 1rem !important;
    padding: 8px !important;
}}

.css-1wrcr25, .css-1vq4p4l {{
    color: white !important;
}}

hr {{
    border: none !important;
    height: 2px !important;
    background: linear-gradient(90deg, transparent, rgba(160, 174, 192, 0.5), transparent) !important;
    margin: 30px 0 !important;
}}

.sidebar-content p, .sidebar-content span, .sidebar-content div {{
    color: white !important;
}}

.stNumberInput > div > div > input {{
    background: rgba(40, 45, 60, 0.9) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
}}

div[data-testid="stMetric"]:nth-of-type(1) {{
    background: linear-gradient(135deg, rgba(160, 174, 192, 0.3) 0%, rgba(113, 128, 150, 0.3) 100%) !important;
    border: 1px solid rgba(160, 174, 192, 0.5) !important;
}}

div[data-testid="stMetric"]:nth-of-type(1) label {{
    color: #E2E8F0 !important;
}}

div[data-testid="stMetric"]:nth-of-type(1) div {{
    color: #CBD5E0 !important;
    text-shadow: 1px 1px 3px rgba(160,174,192,0.3) !important;
}}
</style>
""", unsafe_allow_html=True)

lang = st.sidebar.radio("Language / اللغة", ["English", "العربية"], index=0)

def _(en, ar):
    return en if lang == "English" else ar

@st.cache_data
def load_data():
    files = {}
    try:
        files['main'] = pd.read_csv('fortune500_cleaned.csv')
        st.sidebar.success(f"Main: {len(files['main']):,} rows")
    except:
        files['main'] = pd.DataFrame()
    try:
        files['pred2024'] = pd.read_csv('fortune500_2024_predictions.csv')
        st.sidebar.success(f"2024: {len(files['pred2024']):,} rows")
    except:
        files['pred2024'] = pd.DataFrame()
    try:
        files['models'] = pd.read_csv('fortune500_models_performance.csv')
        st.sidebar.success(f"Models: {len(files['models'])} models")
    except:
        files['models'] = pd.DataFrame()
    try:
        files['test'] = pd.read_csv('fortune500_test_predictions.csv')
        st.sidebar.success(f"Test: {len(files['test']):,} rows")
    except:
        files['test'] = pd.DataFrame()
    return files

data = load_data()
df = data['main']

if df.empty:
    st.error(_("Main data file not found!", "ملف البيانات الرئيسي غير موجود!"))
    st.stop()

df['profit_margin'] = (df['profit_mil'] / df['revenue_mil']) * 100

colors = {
    'primary': '#4A5568',
    'secondary': '#2D3748',
    'accent1': '#A0AEC0',
    'accent2': '#718096',
    'accent3': '#4A5568',
    'success': '#48BB78',
    'danger': '#F56565',
    'warning': '#ECC94B',
    'info': '#A0AEC0',
    'light': '#F7FAFC',
    'dark': '#1A202C',
    'gray1': '#CBD5E0',
    'gray2': '#E2E8F0',
    'gray3': '#EDF2F7'
}

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(45, 55, 72, 0.95) 0%, rgba(26, 32, 44, 0.95) 100%);
            backdrop-filter: blur(12px);
            padding: 40px; 
            border-radius: 25px; 
            margin-bottom: 30px; 
            text-align: center;
            border: 1px solid rgba(255,255,255,0.25);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);">
    <h1 style="color: white; margin: 0; font-size: 3.2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-weight: 700; letter-spacing: 1px;">
        {_('Fortune 500 Analytics Dashboard', 'لوحة تحليل Fortune 500')}
    </h1>
    <p style="color: rgba(255,255,255,0.95); margin-top: 15px; font-size: 1.4rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
        {_('1996-2024 Analysis & Predictions', 'تحليل وتوقعات 1996-2024')}
    </p>
    <div style="background: rgba(160, 174, 192, 0.2); 
                backdrop-filter: blur(8px);
                padding: 15px; 
                border-radius: 15px; 
                margin-top: 25px; 
                border: 1px solid rgba(160, 174, 192, 0.5);
                max-width: 400px;
                margin-left: auto;
                margin-right: auto;">
        <p style="color: white; margin: 0; font-size: 1.2rem; font-weight: 500;">
            {_('Developed by: Mohammad Naser', 'تم التطوير بواسطة: محمد زكريا ناصر')}
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(45, 55, 72, 0.25) 0%, rgba(26, 32, 44, 0.25) 100%);
                backdrop-filter: blur(12px);
                padding: 25px; 
                border-radius: 20px; 
                margin-bottom: 25px;
                border: 1px solid rgba(255,255,255,0.2);">
        <h3 style="color: white; margin-top: 0; font-size: 1.5rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
            {_('Control Panel', 'لوحة التحكم')}
        </h3>
        <p style="color: rgba(255,255,255,0.9); margin-bottom: 0; font-size: 1rem;">
            {_('Developer: Mohammad Naser', 'المطور: محمد زكريا ناصر')}
        </p>
        <p style="color: rgba(255,255,255,0.7); margin-bottom: 0; font-size: 0.9rem;">
            {_('Data Analyst', 'محلل بيانات')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        _("Select Analysis", "اختر التحليل"),
        [
            _("Year Analysis", "تحليل السنوات"),
            _("Company Analysis", "تحليل الشركات"),
            _("Year Comparison", "مقارنة السنوات"),
            _("Predictions & Models", "التوقعات والنماذج"),
            _("Data Overview", "نظرة عامة")
        ]
    )

if menu == _("Year Analysis", "تحليل السنوات"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header(_("Year Analysis", "تحليل السنوات"))
    col1, col2 = st.columns([3,1])
    with col1:
        year = st.selectbox(_("Select Year", "اختر السنة"), sorted(df['year'].unique(), reverse=True))
    with col2:
        top_n = st.number_input(_("Companies", "الشركات"), 5, 50, 15)
    df_year = df[df['year'] == year]
    if not df_year.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(_("Companies", "الشركات"), f"{len(df_year):,}")
        with col2:
            st.metric(_("Total Revenue", "إجمالي الإيرادات"), f"${df_year['revenue_mil'].sum():,.0f}M")
        with col3:
            st.metric(_("Avg Revenue", "متوسط الإيرادات"), f"${df_year['revenue_mil'].mean():,.0f}M")
        with col4:
            st.metric(_("Avg Margin", "متوسط الهامش"), f"{df_year['profit_margin'].mean():.1f}%")
        tabs = st.tabs([_("Top Companies", "أفضل الشركات"), _("Revenue Distribution", "توزيع الإيرادات"), _("Industry Analysis", "تحليل الصناعات")])
        with tabs[0]:
            top = df_year.nlargest(top_n, 'revenue_mil')
            fig = px.bar(top, x='revenue_mil', y='name', orientation='h',
                        title=f"{_('Top', 'أفضل')} {top_n} {_('Companies', 'شركة')} - {year}",
                        color='revenue_mil', color_continuous_scale='gray')
            fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='white', size=12), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top[['rank','name','revenue_mil','profit_mil','profit_margin','industry']], use_container_width=True)
        with tabs[1]:
            fig = px.histogram(df_year, x='revenue_mil', nbins=50, title=_("Revenue Distribution", "توزيع الإيرادات"))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            height=400, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        with tabs[2]:
            ind = df_year.groupby('industry').agg({'revenue_mil':'sum','profit_margin':'mean'}).sort_values('revenue_mil', ascending=False).head(15)
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.bar(ind.reset_index(), x='revenue_mil', y='industry', orientation='h',
                            title=_("Revenue by Industry", "الإيرادات حسب الصناعة"),
                            color='revenue_mil', color_continuous_scale='gray')
                fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                 height=500, font=dict(color='white'), title_font_color='white')
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.bar(ind.reset_index(), x='profit_margin', y='industry', orientation='h',
                            title=_("Margin by Industry", "الهامش حسب الصناعة"),
                            color='profit_margin', color_continuous_scale='gray')
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                 height=500, font=dict(color='white'), title_font_color='white')
                st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == _("Company Analysis", "تحليل الشركات"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header(_("Company Analysis", "تحليل الشركات"))
    company = st.selectbox(_("Select Company", "اختر الشركة"), sorted(df['name'].unique()))
    df_comp = df[df['name'] == company].sort_values('year')
    if not df_comp.empty:
        latest = df_comp.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(_("Years in List", "السنوات في القائمة"), len(df_comp))
        with col2:
            st.metric(_("Latest Revenue", "آخر إيرادات"), f"${latest['revenue_mil']:,.0f}M")
        with col3:
            st.metric(_("Latest Rank", "آخر ترتيب"), f"#{int(latest['rank'])}")
        with col4:
            st.metric(_("Latest Margin", "آخر هامش"), f"{latest['profit_margin']:.1f}%")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.line(df_comp, x='year', y='revenue_mil', title=_("Revenue Trend", "اتجاه الإيرادات"), markers=True)
            fig1.update_traces(line=dict(color='#A0AEC0', width=3), marker=dict(color='#A0AEC0', size=8))
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                             height=400, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.line(df_comp, x='year', y='rank', title=_("Rank Trend", "اتجاه الترتيب"), markers=True)
            fig2.update_traces(line=dict(color='#718096', width=3), marker=dict(color='#718096', size=8))
            fig2.update_yaxes(autorange="reversed")
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                             height=400, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig2, use_container_width=True)
        st.subheader(_("Historical Data", "البيانات التاريخية"))
        st.dataframe(df_comp[['year','rank','revenue_mil','profit_mil','profit_margin']], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == _("Year Comparison", "مقارنة السنوات"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header(_("Year Comparison", "مقارنة السنوات"))
    years = sorted(df['year'].unique(), reverse=True)
    col1, col2 = st.columns(2)
    with col1:
        y1 = st.selectbox(_("First Year", "السنة الأولى"), years, index=3)
    with col2:
        y2 = st.selectbox(_("Second Year", "السنة الثانية"), years, index=0)
    if y1 != y2:
        d1 = df[df['year'] == y1]
        d2 = df[df['year'] == y2]
        rev_growth = ((d2['revenue_mil'].sum() - d1['revenue_mil'].sum()) / d1['revenue_mil'].sum()) * 100
        avg_growth = ((d2['revenue_mil'].mean() - d1['revenue_mil'].mean()) / d1['revenue_mil'].mean()) * 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(_("Revenue Growth", "نمو الإيرادات"), f"{rev_growth:+.1f}%")
        with col2:
            st.metric(_("Avg Growth", "متوسط النمو"), f"{avg_growth:+.1f}%")
        with col3:
            st.metric(_("Companies Change", "تغير الشركات"), f"{len(d2)-len(d1):+d}")
        comp = pd.DataFrame({
            _("Year", "السنة"): [str(y1), str(y2)],
            _("Total Revenue", "إجمالي الإيرادات"): [d1['revenue_mil'].sum(), d2['revenue_mil'].sum()],
            _("Avg Revenue", "متوسط الإيرادات"): [d1['revenue_mil'].mean(), d2['revenue_mil'].mean()],
            _("Companies", "الشركات"): [len(d1), len(d2)]
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name=_("Total Revenue", "إجمالي الإيرادات"), 
                            x=comp[_("Year", "السنة")], y=comp[_("Total Revenue", "إجمالي الإيرادات")],
                            marker_color='#A0AEC0'))
        fig.add_trace(go.Bar(name=_("Avg Revenue", "متوسط الإيرادات"), 
                            x=comp[_("Year", "السنة")], y=comp[_("Avg Revenue", "متوسط الإيرادات")],
                            marker_color='#718096'))
        fig.update_layout(barmode='group', height=400, 
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                         font=dict(color='white', size=12), title_font_color='white',
                         legend_font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == _("Predictions & Models", "التوقعات والنماذج"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header(_("Predictions & AI Models", "التوقعات والنماذج الذكية"))
    
    if not data['pred2024'].empty:
        st.subheader(_("2024 Predictions", "توقعات 2024"))
        df_pred = data['pred2024']
        
        revenue_col = None
        name_col = None
        rank_col = None
        
        for col in df_pred.columns:
            col_lower = col.lower()
            if 'revenue' in col_lower or 'rev' in col_lower or 'pred' in col_lower:
                revenue_col = col
            if 'name' in col_lower or 'company' in col_lower:
                name_col = col
            if 'rank' in col_lower:
                rank_col = col
        
        if revenue_col is None and len(df_pred.select_dtypes(include=[np.number]).columns) > 0:
            revenue_col = df_pred.select_dtypes(include=[np.number]).columns[0]
        
        display_cols = []
        if name_col:
            display_cols.append(name_col)
        if revenue_col:
            display_cols.append(revenue_col)
        if rank_col:
            display_cols.append(rank_col)
        
        if revenue_col and name_col:
            df_pred_sorted = df_pred.sort_values(revenue_col, ascending=False).head(20)
            fig = px.bar(df_pred_sorted, x=revenue_col, y=name_col, orientation='h',
                        title=_("Top 20 Predicted Companies 2024", "أفضل 20 شركة متوقعة 2024"),
                        color=revenue_col, color_continuous_scale='gray')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            height=500, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        
        if display_cols:
            st.dataframe(df_pred[display_cols].head(50), use_container_width=True)
        else:
            st.dataframe(df_pred.head(50), use_container_width=True)
    else:
        st.info(_("2024 predictions file not available", "ملف توقعات 2024 غير متوفر"))
    
    if not data['models'].empty:
        st.subheader(_("Model Performance", "أداء النماذج"))
        df_models = data['models']
        
        model_col = None
        accuracy_col = None
        
        for col in df_models.columns:
            col_lower = col.lower()
            if 'model' in col_lower or 'name' in col_lower:
                model_col = col
            if 'acc' in col_lower or 'score' in col_lower or 'r2' in col_lower:
                accuracy_col = col
        
        if accuracy_col:
            if model_col:
                fig = px.bar(df_models, x=model_col, y=accuracy_col, 
                           title=_("Model Accuracy", "دقة النماذج"),
                           color=accuracy_col, color_continuous_scale='gray')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                height=400, xaxis_tickangle=45, font=dict(color='white'), 
                                title_font_color='white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(df_models, y=accuracy_col, 
                           title=_("Model Accuracy", "دقة النماذج"),
                           color=accuracy_col, color_continuous_scale='gray')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                height=400, font=dict(color='white'), title_font_color='white')
                st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_models, use_container_width=True)
    
    if not data['test'].empty:
        st.subheader(_("Test Predictions", "توقعات الاختبار"))
        df_test = data['test']
        
        actual_col = None
        predicted_col = None
        
        for col in df_test.columns:
            col_lower = col.lower()
            if 'actual' in col_lower or 'true' in col_lower:
                actual_col = col
            if 'pred' in col_lower or 'predict' in col_lower:
                predicted_col = col
        
        if actual_col and predicted_col:
            fig = px.scatter(df_test.head(100), x=actual_col, y=predicted_col,
                           title=_("Actual vs Predicted", "الفعلية مقابل المتوقعة"),
                           labels={actual_col: _("Actual", "فعلية"), predicted_col: _("Predicted", "متوقعة")})
            fig.update_traces(marker=dict(color='#A0AEC0', size=5))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            height=500, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_test.head(50), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header(_("Data Overview", "نظرة عامة"))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(_("Total Years", "إجمالي السنوات"), df['year'].nunique())
    with col2:
        st.metric(_("Unique Companies", "الشركات الفريدة"), df['name'].nunique())
    with col3:
        st.metric(_("Total Revenue", "إجمالي الإيرادات"), f"${df['revenue_mil'].sum()/1000000:,.1f}T")
    with col4:
        st.metric(_("Avg Annual Growth", "متوسط النمو السنوي"), f"{df.groupby('year')['revenue_mil'].mean().pct_change().mean()*100:.1f}%")
    
    yearly = df.groupby('year').agg({'revenue_mil':'mean','profit_mil':'mean','profit_margin':'mean'}).reset_index()
    
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=(
                           _("Average Revenue Trend", "اتجاه متوسط الإيرادات"),
                           _("Average Profit Trend", "اتجاه متوسط الأرباح"),
                           _("Average Margin Trend", "اتجاه متوسط الهامش")
                       ))
    
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['revenue_mil'], 
                            name=_("Revenue","الإيرادات"), line=dict(color='#A0AEC0', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['profit_mil'], 
                            name=_("Profit","الأرباح"), line=dict(color='#48BB78', width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['profit_margin'], 
                            name=_("Margin","الهامش"), line=dict(color='#ECC94B', width=3)), row=3, col=1)
    
    fig.update_layout(height=700, showlegend=True, 
                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                     font=dict(color='white', size=12), title_font_color='white',
                     legend_font_color='white')
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', gridwidth=1, row=1, col=1)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', gridwidth=1, row=1, col=1)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', gridwidth=1, row=2, col=1)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', gridwidth=1, row=2, col=1)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', gridwidth=1, row=3, col=1)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', gridwidth=1, row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    top = df.groupby('name')['revenue_mil'].max().nlargest(15)
    fig2 = px.bar(x=top.values, y=top.index, orientation='h',
                 title=_("Top 15 Companies All Time", "أفضل 15 شركة على الإطلاق"),
                 color=top.values, color_continuous_scale='gray')
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                      height=500, font=dict(color='white', size=12), title_font_color='white')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(45, 55, 72, 0.9) 0%, rgba(26, 32, 44, 0.9) 100%);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 30px;
            margin-top: 40px;
            border: 1px solid rgba(255,255,255,0.2);
            text-align: center;">
    <p style="color: white; font-size: 1.3rem; margin-bottom: 15px; font-weight: 600; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
        <strong>{_('Fortune 500 Analytics Dashboard', 'لوحة تحليل Fortune 500')}</strong>
    </p>
    <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 15px; flex-wrap: wrap;">
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem;">
            {_('Developed by: Mohammad Naser', 'تم التطوير بواسطة: محمد زكريا ناصر')}
        </p>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem;">
            {_('Data Analyst', 'محلل بيانات')}
        </p>
    </div>
    <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 15px; flex-wrap: wrap;">
        <p style="color: rgba(255,255,255,0.8); font-size: 0.95rem;">
            1996-2024
        </p>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.95rem;">
            {_('Powered by Streamlit & Plotly', 'بتقنية Streamlit و Plotly')}
        </p>
    </div>
    <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 10px;">
        © 2024 {_('All Rights Reserved', 'جميع الحقوق محفوظة')}
    </p>
</div>
""", unsafe_allow_html=True)
