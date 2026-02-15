import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import warnings
import os
from datetime import datetime
from PIL import Image
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fortune 500 Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_of_image(image_path):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

image_path = r"WhatsApp Image 2026-02-11 at 3.32.24 PM.jpeg"
if os.path.exists(image_path):
    image_base64 = get_base64_of_image(image_path)
else:
    image_base64 = None

if image_base64:
    bg_style = f"""
    .stApp {{
        background-image: url("data:image/jpeg;base64,{image_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    """
else:
    bg_style = """
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    """

st.markdown(f"""
<style>
{bg_style}

.main > div {{
    background: transparent !important;
    backdrop-filter: none !important;
}}

.css-1d391kg, .css-163ttbj, [data-testid="stSidebar"] > div:first-child {{
    background: rgba(10, 10, 20, 0.85) !important;
    backdrop-filter: blur(10px) !important;
    border-right: 1px solid rgba(255,255,255,0.15) !important;
}}

.intro-header {{
    background: linear-gradient(135deg, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.4) 100%);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 30px;
    padding: 50px 40px;
    margin: 20px 0 40px 0;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}}

.intro-header h1 {{
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: 2px;
}}

.intro-header p {{
    color: rgba(255,255,255,0.95);
    font-size: 1.5rem;
    margin-top: 15px;
    font-weight: 300;
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
}}

.stButton > button {{
    background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    transition: all 0.3s ease !important;
}}

.stButton > button:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 20px rgba(74, 85, 104, 0.4) !important;
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
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%) !important;
}}

.stSelectbox label, .stDropdown label {{
    color: #ffffff !important;
    font-weight: 600 !important;
}}

.stSelectbox > div > div {{
    background: rgba(40, 45, 60, 0.9) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
}}

h1, h2, h3, h4, h5, h6 {{
    color: #ffffff !important;
}}

.stMarkdown p, .stMarkdown span {{
    color: rgba(255,255,255,0.95) !important;
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
}}

.stMetric div {{
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
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
}}

.stDataFrame th {{
    background: rgba(74, 85, 104, 0.3) !important;
    color: white !important;
    font-weight: 600 !important;
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

.stNumberInput > div > div > input {{
    background: rgba(40, 45, 60, 0.9) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
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
        st.sidebar.success(f"✅ Main: {len(files['main']):,} rows")
    except:
        files['main'] = pd.DataFrame()
        st.sidebar.warning("⚠️ Main data file not found")
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

with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(45, 55, 72, 0.3) 0%, rgba(26, 32, 44, 0.3) 100%);
                backdrop-filter: blur(12px);
                padding: 25px; 
                border-radius: 20px; 
                margin-bottom: 25px;
                border: 1px solid rgba(255,255,255,0.2);
                text-align: center;">
        <h3 style="color: white; margin: 0 0 15px 0; font-size: 1.5rem;">⚡ CONTROL PANEL</h3>
        <hr style="margin: 15px 0; opacity: 0.3;">
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        _("Select Analysis", "اختر التحليل"),
        [
            _("📅 Year Analysis", "📅 تحليل السنوات"),
            _("🏢 Company Analysis", "🏢 تحليل الشركات"),
            _("📈 Year Comparison", "📈 مقارنة السنوات"),
            _("🤖 Predictions & Models", "🤖 التوقعات والنماذج"),
            _("📋 Data Overview", "📋 نظرة عامة"),
            _("📸 About", "📸 حول")  # تم تغيير الاسم
        ]
    )

data = load_data()
df = data['main']

# تم إزالة الـ intro-header بالكامل

if df.empty:
    st.error(_("⚠️ Main data file not found! Please check if fortune500_cleaned.csv exists.", 
               "⚠️ ملف البيانات الرئيسي غير موجود! الرجاء التأكد من وجود الملف."))
    st.stop()

df['profit_margin'] = (df['profit_mil'] / df['revenue_mil']) * 100
df['revenue_bil'] = df['revenue_mil'] / 1000
df['profit_bil'] = df['profit_mil'] / 1000

# ==================== YEAR ANALYSIS ====================
if menu == _("📅 Year Analysis", "📅 تحليل السنوات"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header(_("📅 Year Analysis", "📅 تحليل السنوات"))
    
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
            st.metric(_("Total Revenue", "إجمالي الإيرادات"), f"${df_year['revenue_bil'].sum():,.1f}B")
        with col3:
            st.metric(_("Avg Revenue", "متوسط الإيرادات"), f"${df_year['revenue_bil'].mean():,.1f}B")
        with col4:
            st.metric(_("Avg Margin", "متوسط الهامش"), f"{df_year['profit_margin'].mean():.1f}%")
        
        tabs = st.tabs([
            _("🏆 Top Companies", "🏆 أفضل الشركات"), 
            _("📊 Distribution", "📊 التوزيع"), 
            _("🏭 Industries", "🏭 الصناعات")
        ])
        
        with tabs[0]:
            top = df_year.nlargest(top_n, 'revenue_mil')
            fig = px.bar(top, x='revenue_bil', y='name', orientation='h',
                        title=f"{_('Top', 'أفضل')} {top_n} {_('Companies', 'شركة')} - {year}",
                        color='revenue_bil', color_continuous_scale='viridis')
            fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='white', size=12), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top[['rank','name','revenue_bil','profit_bil','profit_margin','industry']], 
                        use_container_width=True)
        
        with tabs[1]:
            fig = px.histogram(df_year, x='revenue_bil', nbins=50, 
                              title=_("Revenue Distribution (Billions $)", "توزيع الإيرادات (بالمليارات)"))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            height=400, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            ind = df_year.groupby('industry').agg({
                'revenue_bil': 'sum',
                'profit_margin': 'mean'
            }).sort_values('revenue_bil', ascending=False).head(15)
            
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.bar(ind.reset_index(), x='revenue_bil', y='industry', orientation='h',
                            title=_("Revenue by Industry (B$)", "الإيرادات حسب الصناعة"),
                            color='revenue_bil', color_continuous_scale='viridis')
                fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                 height=500, font=dict(color='white'), title_font_color='white')
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.bar(ind.reset_index(), x='profit_margin', y='industry', orientation='h',
                            title=_("Margin by Industry", "الهامش حسب الصناعة"),
                            color='profit_margin', color_continuous_scale='viridis')
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                 height=500, font=dict(color='white'), title_font_color='white')
                st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== COMPANY ANALYSIS ====================
elif menu == _("🏢 Company Analysis", "🏢 تحليل الشركات"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header(_("🏢 Company Analysis", "🏢 تحليل الشركات"))
    
    company = st.selectbox(_("Select Company", "اختر الشركة"), sorted(df['name'].unique()))
    df_comp = df[df['name'] == company].sort_values('year')
    
    if not df_comp.empty:
        latest = df_comp.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(_("Years in List", "السنوات في القائمة"), len(df_comp))
        with col2:
            st.metric(_("Latest Revenue", "آخر إيرادات"), f"${latest['revenue_bil']:,.1f}B")
        with col3:
            st.metric(_("Latest Rank", "آخر ترتيب"), f"#{int(latest['rank'])}")
        with col4:
            st.metric(_("Latest Margin", "آخر هامش"), f"{latest['profit_margin']:.1f}%")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.line(df_comp, x='year', y='revenue_bil', 
                          title=_("Revenue Trend (Billions $)", "اتجاه الإيرادات (بالمليارات)"), 
                          markers=True)
            fig1.update_traces(line=dict(color='#4A5568', width=3), 
                              marker=dict(color='#4A5568', size=8))
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                             height=400, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.line(df_comp, x='year', y='rank', 
                          title=_("Rank Trend", "اتجاه الترتيب"), 
                          markers=True)
            fig2.update_traces(line=dict(color='#718096', width=3), 
                              marker=dict(color='#718096', size=8))
            fig2.update_yaxes(autorange="reversed")
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                             height=400, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader(_("Historical Data", "البيانات التاريخية"))
        st.dataframe(df_comp[['year','rank','revenue_bil','profit_bil','profit_margin']], 
                    use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== YEAR COMPARISON ====================
elif menu == _("📈 Year Comparison", "📈 مقارنة السنوات"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header(_("📈 Year Comparison", "📈 مقارنة السنوات"))
    
    years = sorted(df['year'].unique(), reverse=True)
    col1, col2 = st.columns(2)
    with col1:
        y1 = st.selectbox(_("First Year", "السنة الأولى"), years, index=3)
    with col2:
        y2 = st.selectbox(_("Second Year", "السنة الثانية"), years, index=0)
    
    if y1 != y2:
        d1 = df[df['year'] == y1]
        d2 = df[df['year'] == y2]
        
        rev_growth = ((d2['revenue_bil'].sum() - d1['revenue_bil'].sum()) / d1['revenue_bil'].sum()) * 100
        avg_growth = ((d2['revenue_bil'].mean() - d1['revenue_bil'].mean()) / d1['revenue_bil'].mean()) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(_("Revenue Growth", "نمو الإيرادات"), f"{rev_growth:+.1f}%")
        with col2:
            st.metric(_("Avg Growth", "متوسط النمو"), f"{avg_growth:+.1f}%")
        with col3:
            st.metric(_("Companies Change", "تغير الشركات"), f"{len(d2)-len(d1):+d}")
        
        comp = pd.DataFrame({
            _("Year", "السنة"): [str(y1), str(y2)],
            _("Total Revenue (B$)", "إجمالي الإيرادات"): [d1['revenue_bil'].sum(), d2['revenue_bil'].sum()],
            _("Avg Revenue (B$)", "متوسط الإيرادات"): [d1['revenue_bil'].mean(), d2['revenue_bil'].mean()],
            _("Companies", "الشركات"): [len(d1), len(d2)]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name=_("Total Revenue", "إجمالي الإيرادات"), 
                            x=comp[_("Year", "السنة")], y=comp[_("Total Revenue (B$)", "إجمالي الإيرادات")],
                            marker_color='#4A5568'))
        fig.add_trace(go.Bar(name=_("Avg Revenue", "متوسط الإيرادات"), 
                            x=comp[_("Year", "السنة")], y=comp[_("Avg Revenue (B$)", "متوسط الإيرادات")],
                            marker_color='#718096'))
        fig.update_layout(barmode='group', height=400, 
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                         font=dict(color='white', size=12), title_font_color='white',
                         legend_font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PREDICTIONS & MODELS ====================
elif menu == _("🤖 Predictions & Models", "🤖 التوقعات والنماذج"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header(_("🤖 Predictions & AI Models", "🤖 التوقعات والنماذج الذكية"))
    
    if not data['pred2024'].empty:
        st.subheader(_("📊 2024 Predictions", "📊 توقعات 2024"))
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
                        color=revenue_col, color_continuous_scale='viridis')
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
        st.subheader(_("📈 Model Performance", "📈 أداء النماذج"))
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
                           color=accuracy_col, color_continuous_scale='viridis')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                height=400, xaxis_tickangle=45, font=dict(color='white'), 
                                title_font_color='white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(df_models, y=accuracy_col, 
                           title=_("Model Accuracy", "دقة النماذج"),
                           color=accuracy_col, color_continuous_scale='viridis')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                height=400, font=dict(color='white'), title_font_color='white')
                st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_models, use_container_width=True)
    
    if not data['test'].empty:
        st.subheader(_("🧪 Test Predictions", "🧪 توقعات الاختبار"))
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
            fig.update_traces(marker=dict(color='#4A5568', size=5))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                            height=500, font=dict(color='white'), title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_test.head(50), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== DATA OVERVIEW ====================
elif menu == _("📋 Data Overview", "📋 نظرة عامة"):
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header(_("📋 Data Overview", "📋 نظرة عامة"))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(_("Total Years", "إجمالي السنوات"), df['year'].nunique())
    with col2:
        st.metric(_("Unique Companies", "الشركات الفريدة"), df['name'].nunique())
    with col3:
        st.metric(_("Total Revenue", "إجمالي الإيرادات"), f"${df['revenue_bil'].sum()/1000:,.1f}T")
    with col4:
        avg_growth = df.groupby('year')['revenue_bil'].mean().pct_change().mean() * 100
        st.metric(_("Avg Annual Growth", "متوسط النمو السنوي"), f"{avg_growth:.1f}%")
    
    yearly = df.groupby('year').agg({
        'revenue_bil': 'mean',
        'profit_bil': 'mean',
        'profit_margin': 'mean'
    }).reset_index()
    
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=(
                           _("Average Revenue Trend (B$)", "اتجاه متوسط الإيرادات"),
                           _("Average Profit Trend (B$)", "اتجاه متوسط الأرباح"),
                           _("Average Margin Trend", "اتجاه متوسط الهامش")
                       ))
    
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['revenue_bil'], 
                            name=_("Revenue","الإيرادات"), line=dict(color='#4A5568', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['profit_bil'], 
                            name=_("Profit","الأرباح"), line=dict(color='#48BB78', width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['profit_margin'], 
                            name=_("Margin","الهامش"), line=dict(color='#ECC94B', width=3)), row=3, col=1)
    
    fig.update_layout(height=700, showlegend=True, 
                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                     font=dict(color='white', size=12), title_font_color='white',
                     legend_font_color='white')
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', gridwidth=1)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', gridwidth=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    top = df.groupby('name')['revenue_bil'].max().nlargest(15)
    fig2 = px.bar(x=top.values, y=top.index, orientation='h',
                 title=_("Top 15 Companies All Time", "أفضل 15 شركة على الإطلاق"),
                 color=top.values, color_continuous_scale='viridis')
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                      height=500, font=dict(color='white', size=12), title_font_color='white')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== ABOUT (المطور في آخر صفحة فقط) ====================
else:  # menu == "📸 About"
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    st.header(_("📸 About Developer", "📸 عن المطور"))
    
    # عمودين: صورة رمزية ومعلومات المطور
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%); width: 200px; height: 200px; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; border: 4px solid #A0AEC0; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
                <span style="font-size: 5rem;">👨‍💻</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # إضافة مكان لرفع صورة شخصية
        st.markdown("### 📸 Profile Picture")
        uploaded_file = st.file_uploader(
            _("Upload your image...", "ارفع صورتك..."), 
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=_("Your Image", "صورتك"), width=150)
    
    with col2:
        st.markdown(f"""
        <div style="padding: 20px;">
            <h2 style="color: white; margin-bottom: 20px; font-size: 2.5rem;">{_('Developer Information', 'معلومات المطور')}</h2>
            <p style="color: #A0AEC0; font-size: 1.8rem; margin-bottom: 20px;">
                <strong style="color: white;">Mohammad Zakaria Naser</strong>
            </p>
            <p style="color: white; margin-bottom: 15px; font-size: 1.2rem;">
                {_('This Fortune 500 Analytics Dashboard was developed from scratch by Mohammad Naser in 2024.', 'تم تطوير لوحة تحليل Fortune 500 هذه من الصفر بواسطة محمد ناصر في 2024.')}
            </p>
            <p style="color: white; margin-bottom: 15px; font-size: 1.2rem;">
                {_('All code, design, and functionality are original work.', 'جميع الكود والتصميم والوظائف هي عمل أصلي.')}
            </p>
            <div style="background: rgba(160, 174, 192, 0.1); border-radius: 20px; padding: 25px; margin-top: 25px; border: 1px solid rgba(255,255,255,0.1);">
                <p style="color: #A0AEC0; margin: 10px 0; font-size: 1.1rem;">📧 mohammad.naser@email.com</p>
                <p style="color: #A0AEC0; margin: 10px 0; font-size: 1.1rem;">🔗 github.com/mohammadnaser</p>
                <p style="color: #A0AEC0; margin: 10px 0; font-size: 1.1rem;">📅 2024</p>
                <p style="color: #A0AEC0; margin: 10px 0; font-size: 1.1rem;">📍 Palestine</p>
            </div>
            <hr style="margin: 30px 0; border-color: rgba(255,255,255,0.1);">
            <p style="color: #A0AEC0; text-align: center; font-size: 1.1rem;">
                © 2024 {_('All Rights Reserved.', 'جميع الحقوق محفوظة.')}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # تذييل بسيط بدون ذكر المطور
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.4) 100%);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 20px;
                margin-top: 20px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.2);">
        <p style="color: white; font-weight: 600;">FORTUNE 500 EXECUTIVE ANALYTICS</p>
    </div>
    """, unsafe_allow_html=True)
