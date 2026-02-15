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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# تحميل الصور
background_image_path = r"WhatsApp Image 2026-02-11 at 3.32.24 PM.jpeg"
profile_image_path = r"WhatsApp Image 2026-02-10 at 1.34.39 PM.jpeg"  # صورتك الشخصية

background_image_base64 = get_base64_of_image(background_image_path)
profile_image_base64 = get_base64_of_image(profile_image_path)

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{background_image_base64}");
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

/* تنسيق صورة المطور في الشريط الجانبي */
.developer-profile {{
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, rgba(45, 55, 72, 0.3) 0%, rgba(26, 32, 44, 0.3) 100%);
    border-radius: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(5px);
    animation: fadeIn 1s ease;
}}

.developer-image {{
    width: 120px;
    height: 120px;
    border-radius: 50%;
    margin: 0 auto 15px;
    border: 3px solid #A0AEC0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
    object-fit: cover;
}}

.developer-image:hover {{
    transform: scale(1.05);
    border-color: white;
    box-shadow: 0 12px 25px rgba(160, 174, 192, 0.4);
}}

.developer-name {{
    color: white;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 5px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}}

.developer-title {{
    color: #A0AEC0;
    font-size: 0.9rem;
    margin-bottom: 10px;
    letter-spacing: 1px;
}}

.developer-badge {{
    display: inline-block;
    background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin-top: 10px;
    border: 1px solid rgba(255,255,255,0.2);
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(-20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
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

/* تنسيق للشريط الجانبي */
.sidebar-section {{
    background: rgba(20, 25, 40, 0.4);
    backdrop-filter: blur(5px);
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}}

.sidebar-title {{
    color: white;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 10px;
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    # صورة المdeveloper - تظهر في أعلى الشريط الجانبي
    st.markdown(f"""
    <div class="developer-profile">
        <img src="data:image/jpeg;base64,{profile_image_base64}" class="developer-image" alt="Developer">
        <div class="developer-name">محمد زكريا ناصر</div>
        <div class="developer-title">Mohammad Naser</div>
        <div style="color: #A0AEC0; font-size: 0.8rem; margin-top: 5px;">
            ⭐ {('Data Analyst', 'محلل بيانات')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # LANGUAGE SELECTION (بدون استخدام دالة الترجمة)
    lang = st.radio("Language / اللغة", ["English", "العربية"], index=0, key="language")
    
    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
    
    # Control Panel
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-title">{'🎮 Control Panel' if lang == 'English' else '🎮 لوحة التحكم'}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # SELECT ANALYSIS (بدون استخدام دالة الترجمة)
    if lang == "English":
        menu_options = [
            "📊 Year Analysis",
            "🏢 Company Analysis",
            "⚖️ Year Comparison",
            "🤖 Predictions & Models",
            "📈 Data Overview"
        ]
    else:
        menu_options = [
            "📊 تحليل السنوات",
            "🏢 تحليل الشركات",
            "⚖️ مقارنة السنوات",
            "🤖 التوقعات والنماذج",
            "📈 نظرة عامة"
        ]
    
    menu = st.radio(
        "Select Analysis" if lang == "English" else "اختر التحليل",
        menu_options,
        key="analysis_menu"
    )
    
    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
    
    # Quick Stats
    with st.expander("📊 Quick Stats" if lang == "English" else "📊 إحصائيات سريعة", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Companies" if lang == "English" else "الشركات", f"{df['name'].nunique():,}" if 'df' in locals() else "0")
        with col2:
            st.metric("Years" if lang == "English" else "السنوات", f"{df['year'].nunique()}" if 'df' in locals() else "0")

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    files = {}
    try:
        files['main'] = pd.read_csv('fortune500_cleaned.csv')
        st.sidebar.success(f"✅ Main: {len(files['main']):,} rows")
    except:
        files['main'] = pd.DataFrame()
    try:
        files['pred2024'] = pd.read_csv('fortune500_2024_predictions.csv')
        st.sidebar.success(f"✅ 2024: {len(files['pred2024']):,} rows")
    except:
        files['pred2024'] = pd.DataFrame()
    try:
        files['models'] = pd.read_csv('fortune500_models_performance.csv')
        st.sidebar.success(f"✅ Models: {len(files['models'])} models")
    except:
        files['models'] = pd.DataFrame()
    try:
        files['test'] = pd.read_csv('fortune500_test_predictions.csv')
        st.sidebar.success(f"✅ Test: {len(files['test']):,} rows")
    except:
        files['test'] = pd.DataFrame()
    return files

data = load_data()
df = data['main']

if df.empty:
    st.error("Main data file not found!" if lang == "English" else "ملف البيانات الرئيسي غير موجود!")
    st.stop()

df['profit_margin'] = (df['profit_mil'] / df['revenue_mil']) * 100

# ==================== MAIN HEADER ====================
st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(45, 55, 72, 0.95) 0%, rgba(26, 32, 44, 0.95) 100%);
            backdrop-filter: blur(12px);
            padding: 25px; 
            border-radius: 20px; 
            margin-bottom: 150px; 
            text-align: center;
            border: 1px solid rgba(255,255,255,0.25);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);">
    <h1 style="color: white; margin: 0; font-size: 3.2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-weight: 700; letter-spacing: 1px;">
        {'Fortune 500 Analytics Dashboard' if lang == 'English' else 'لوحة تحليل Fortune 500'}
    </h1>
    <p style="color: rgba(255,255,255,0.95); margin-top: 15px; font-size: 1.4rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
        {'1996-2024 Analysis & Predictions' if lang == 'English' else 'تحليل وتوقعات 1996-2024'}
    </p>
    <p style="color: #A0AEC0; margin-top: 10px; font-size: 1rem;">
        {'Developed by Mohammad Naser' if lang == 'English' else 'تم التطوير بواسطة محمد ناصر'}
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== MAIN CONTENT BASED ON SELECTION ====================
if menu == "📊 Year Analysis" or menu == "📊 تحليل السنوات":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("📊 Year Analysis" if lang == "English" else "📊 تحليل السنوات")
    col1, col2 = st.columns([3,1])
    with col1:
        year = st.selectbox("Select Year" if lang == "English" else "اختر السنة", sorted(df['year'].unique(), reverse=True))
    with col2:
        top_n = st.number_input("Companies" if lang == "English" else "الشركات", 5, 50, 15)
    df_year = df[df['year'] == year]
    if not df_year.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Companies" if lang == "English" else "الشركات", f"{len(df_year):,}")
        with col2:
            st.metric("Total Revenue" if lang == "English" else "إجمالي الإيرادات", f"${df_year['revenue_mil'].sum():,.0f}M")
        with col3:
            st.metric("Avg Revenue" if lang == "English" else "متوسط الإيرادات", f"${df_year['revenue_mil'].mean():,.0f}M")
        with col4:
            st.metric("Avg Margin" if lang == "English" else "متوسط الهامش", f"{df_year['profit_margin'].mean():.1f}%")
        
        tabs = st.tabs([
            "Top Companies" if lang == "English" else "أفضل الشركات",
            "Revenue Distribution" if lang == "English" else "توزيع الإيرادات",
            "Industry Analysis" if lang == "English" else "تحليل الصناعات"
        ])
        
        with tabs[0]:
            top = df_year.nlargest(top_n, 'revenue_mil')
            fig = px.bar(top, x='revenue_mil', y='name', orientation='h',
                        title=f"{'Top' if lang == 'English' else 'أفضل'} {top_n} {'Companies' if lang == 'English' else 'شركة'} - {year}",
                        color='revenue_mil', color_continuous_scale='gray')
            fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='white', size=12), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top[['rank','name','revenue_mil','profit_mil','profit_margin','industry']], use_container_width=True)
        
        with tabs[1]:
            fig = px.histogram(df_year, x='revenue_mil', nbins=50, 
                             title="Revenue Distribution" if lang == "English" else "توزيع الإيرادات")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            height=400, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            ind = df_year.groupby('industry').agg({'revenue_mil':'sum','profit_margin':'mean'}).sort_values('revenue_mil', ascending=False).head(15)
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.bar(ind.reset_index(), x='revenue_mil', y='industry', orientation='h',
                            title="Revenue by Industry" if lang == "English" else "الإيرادات حسب الصناعة",
                            color='revenue_mil', color_continuous_scale='gray')
                fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                 height=500, font=dict(color='white'), title_font_color='white')
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.bar(ind.reset_index(), x='profit_margin', y='industry', orientation='h',
                            title="Margin by Industry" if lang == "English" else "الهامش حسب الصناعة",
                            color='profit_margin', color_continuous_scale='gray')
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                 height=500, font=dict(color='white'), title_font_color='white')
                st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "🏢 Company Analysis" or menu == "🏢 تحليل الشركات":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("🏢 Company Analysis" if lang == "English" else "🏢 تحليل الشركات")
    company = st.selectbox("Select Company" if lang == "English" else "اختر الشركة", sorted(df['name'].unique()))
    df_comp = df[df['name'] == company].sort_values('year')
    if not df_comp.empty:
        latest = df_comp.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Years in List" if lang == "English" else "السنوات في القائمة", len(df_comp))
        with col2:
            st.metric("Latest Revenue" if lang == "English" else "آخر إيرادات", f"${latest['revenue_mil']:,.0f}M")
        with col3:
            st.metric("Latest Rank" if lang == "English" else "آخر ترتيب", f"#{int(latest['rank'])}")
        with col4:
            st.metric("Latest Margin" if lang == "English" else "آخر هامش", f"{latest['profit_margin']:.1f}%")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.line(df_comp, x='year', y='revenue_mil', 
                         title="Revenue Trend" if lang == "English" else "اتجاه الإيرادات", markers=True)
            fig1.update_traces(line=dict(color='#A0AEC0', width=3), marker=dict(color='#A0AEC0', size=8))
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                             height=400, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.line(df_comp, x='year', y='rank', 
                         title="Rank Trend" if lang == "English" else "اتجاه الترتيب", markers=True)
            fig2.update_traces(line=dict(color='#718096', width=3), marker=dict(color='#718096', size=8))
            fig2.update_yaxes(autorange="reversed")
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                             height=400, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Historical Data" if lang == "English" else "البيانات التاريخية")
        st.dataframe(df_comp[['year','rank','revenue_mil','profit_mil','profit_margin']], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "⚖️ Year Comparison" or menu == "⚖️ مقارنة السنوات":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("⚖️ Year Comparison" if lang == "English" else "⚖️ مقارنة السنوات")
    years = sorted(df['year'].unique(), reverse=True)
    col1, col2 = st.columns(2)
    with col1:
        y1 = st.selectbox("First Year" if lang == "English" else "السنة الأولى", years, index=3)
    with col2:
        y2 = st.selectbox("Second Year" if lang == "English" else "السنة الثانية", years, index=0)
    
    if y1 != y2:
        d1 = df[df['year'] == y1]
        d2 = df[df['year'] == y2]
        rev_growth = ((d2['revenue_mil'].sum() - d1['revenue_mil'].sum()) / d1['revenue_mil'].sum()) * 100
        avg_growth = ((d2['revenue_mil'].mean() - d1['revenue_mil'].mean()) / d1['revenue_mil'].mean()) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Revenue Growth" if lang == "English" else "نمو الإيرادات", f"{rev_growth:+.1f}%")
        with col2:
            st.metric("Avg Growth" if lang == "English" else "متوسط النمو", f"{avg_growth:+.1f}%")
        with col3:
            st.metric("Companies Change" if lang == "English" else "تغير الشركات", f"{len(d2)-len(d1):+d}")
        
        comp = pd.DataFrame({
            "Year" if lang == "English" else "السنة": [str(y1), str(y2)],
            "Total Revenue" if lang == "English" else "إجمالي الإيرادات": [d1['revenue_mil'].sum(), d2['revenue_mil'].sum()],
            "Avg Revenue" if lang == "English" else "متوسط الإيرادات": [d1['revenue_mil'].mean(), d2['revenue_mil'].mean()],
            "Companies" if lang == "English" else "الشركات": [len(d1), len(d2)]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Total Revenue" if lang == "English" else "إجمالي الإيرادات", 
                            x=comp["Year" if lang == "English" else "السنة"], 
                            y=comp["Total Revenue" if lang == "English" else "إجمالي الإيرادات"],
                            marker_color='#A0AEC0'))
        fig.add_trace(go.Bar(name="Avg Revenue" if lang == "English" else "متوسط الإيرادات", 
                            x=comp["Year" if lang == "English" else "السنة"], 
                            y=comp["Avg Revenue" if lang == "English" else "متوسط الإيرادات"],
                            marker_color='#718096'))
        fig.update_layout(barmode='group', height=400, 
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                         font=dict(color='white', size=12), title_font_color='white',
                         legend_font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "🤖 Predictions & Models" or menu == "🤖 التوقعات والنماذج":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("🤖 Predictions & AI Models" if lang == "English" else "🤖 التوقعات والنماذج الذكية")
    
    if not data['pred2024'].empty:
        st.subheader("2024 Predictions" if lang == "English" else "توقعات 2024")
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
                        title="Top 20 Predicted Companies 2024" if lang == "English" else "أفضل 20 شركة متوقعة 2024",
                        color=revenue_col, color_continuous_scale='gray')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            height=500, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        
        if display_cols:
            st.dataframe(df_pred[display_cols].head(50), use_container_width=True)
        else:
            st.dataframe(df_pred.head(50), use_container_width=True)
    else:
        st.info("2024 predictions file not available" if lang == "English" else "ملف توقعات 2024 غير متوفر")
    
    if not data['models'].empty:
        st.subheader("Model Performance" if lang == "English" else "أداء النماذج")
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
                           title="Model Accuracy" if lang == "English" else "دقة النماذج",
                           color=accuracy_col, color_continuous_scale='gray')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                height=400, xaxis_tickangle=45, font=dict(color='white'), 
                                title_font_color='white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(df_models, y=accuracy_col, 
                           title="Model Accuracy" if lang == "English" else "دقة النماذج",
                           color=accuracy_col, color_continuous_scale='gray')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                height=400, font=dict(color='white'), title_font_color='white')
                st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_models, use_container_width=True)
    
    if not data['test'].empty:
        st.subheader("Test Predictions" if lang == "English" else "توقعات الاختبار")
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
                           title="Actual vs Predicted" if lang == "English" else "الفعلية مقابل المتوقعة",
                           labels={actual_col: "Actual" if lang == "English" else "فعلية", 
                                  predicted_col: "Predicted" if lang == "English" else "متوقعة"})
            fig.update_traces(marker=dict(color='#A0AEC0', size=5))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            height=500, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_test.head(50), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:  # Data Overview
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("📈 Data Overview" if lang == "English" else "📈 نظرة عامة")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Years" if lang == "English" else "إجمالي السنوات", df['year'].nunique())
    with col2:
        st.metric("Unique Companies" if lang == "English" else "الشركات الفريدة", df['name'].nunique())
    with col3:
        st.metric("Total Revenue" if lang == "English" else "إجمالي الإيرادات", f"${df['revenue_mil'].sum()/1000000:,.1f}T")
    with col4:
        st.metric("Avg Annual Growth" if lang == "English" else "متوسط النمو السنوي", 
                 f"{df.groupby('year')['revenue_mil'].mean().pct_change().mean()*100:.1f}%")
    
    yearly = df.groupby('year').agg({'revenue_mil':'mean','profit_mil':'mean','profit_margin':'mean'}).reset_index()
    
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=(
                           "Average Revenue Trend" if lang == "English" else "اتجاه متوسط الإيرادات",
                           "Average Profit Trend" if lang == "English" else "اتجاه متوسط الأرباح",
                           "Average Margin Trend" if lang == "English" else "اتجاه متوسط الهامش"
                       ))
    
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['revenue_mil'], 
                            name="Revenue" if lang == "English" else "الإيرادات", 
                            line=dict(color='#A0AEC0', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['profit_mil'], 
                            name="Profit" if lang == "English" else "الأرباح", 
                            line=dict(color='#48BB78', width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['profit_margin'], 
                            name="Margin" if lang == "English" else "الهامش", 
                            line=dict(color='#ECC94B', width=3)), row=3, col=1)
    
    fig.update_layout(height=700, showlegend=True, 
                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                     font=dict(color='white', size=12), title_font_color='white',
                     legend_font_color='white')
    
    st.plotly_chart(fig, use_container_width=True)
    
    top = df.groupby('name')['revenue_mil'].max().nlargest(15)
    fig2 = px.bar(x=top.values, y=top.index, orientation='h',
                 title="Top 15 Companies All Time" if lang == "English" else "أفضل 15 شركة على الإطلاق",
                 color=top.values, color_continuous_scale='gray')
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                      height=500, font=dict(color='white', size=12), title_font_color='white')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(45, 55, 72, 0.9) 0%, rgba(26, 32, 44, 0.9) 100%);
            backdrop-filter: blur(12px);
            border-radius: 30px;
            padding: 10px;
            margin-top: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            text-align: center;">
    <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 10px;">
        © 2024 {'All Rights Reserved' if lang == 'English' else 'جميع الحقوق محفوظة'}
    </p>
</div>
""", unsafe_allow_html=True)
