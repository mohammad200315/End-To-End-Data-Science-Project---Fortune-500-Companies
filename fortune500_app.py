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
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

image_path = r"WhatsApp Image 2026-02-11 at 3.32.24 PM.jpeg"
image_base64 = get_base64_of_image(image_path)

if image_base64:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{image_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
.main > div {
    background: rgba(0, 0, 0, 0.65) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    margin: 10px !important;
}

.css-1d391kg, .css-163ttbj, [data-testid="stSidebar"] > div:first-child {
    background: rgba(10, 10, 20, 0.85) !important;
    backdrop-filter: blur(10px) !important;
    border-right: 1px solid rgba(255,255,255,0.15) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

.stMarkdown p, .stMarkdown span {
    color: rgba(255,255,255,0.95) !important;
}

.stMetric {
    background: rgba(30, 35, 50, 0.7) !important;
    backdrop-filter: blur(8px) !important;
    padding: 20px !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}

.stMetric label {
    color: rgba(255,255,255,0.9) !important;
}

.stMetric div {
    color: #ffffff !important;
}

.stSelectbox label {
    color: #ffffff !important;
}

.stDataFrame {
    background: rgba(30, 35, 50, 0.8) !important;
    border-radius: 12px !important;
}

.stDataFrame td, .stDataFrame th {
    color: #ffffff !important;
}

.stRadio label {
    color: white !important;
}
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

st.markdown(f"""
<div style="background: rgba(45, 55, 72, 0.95); padding: 30px; border-radius: 20px; margin-bottom: 30px; text-align: center;">
    <h1 style="color: white; margin: 0; font-size: 3rem;">
        {_('Fortune 500 Analytics Dashboard', 'لوحة تحليل Fortune 500')}
    </h1>
    <p style="color: white; margin-top: 10px; font-size: 1.2rem;">
        {_('1996-2024 Analysis & Predictions', 'تحليل وتوقعات 1996-2024')}
    </p>
    <p style="color: #A0AEC0; margin-top: 15px; font-size: 1rem;">
        {_('Developed by: Mohammad Naser', 'تم التطوير بواسطة: محمد زكريا ناصر')}
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="background: rgba(45, 55, 72, 0.3); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">Control Panel</h3>
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
    st.markdown('<div style="background: rgba(20, 25, 40, 0.7); border-radius: 20px; padding: 20px;">', unsafe_allow_html=True)
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
    st.markdown('<div style="background: rgba(20, 25, 40, 0.7); border-radius: 20px; padding: 20px;">', unsafe_allow_html=True)
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
    st.markdown('<div style="background: rgba(20, 25, 40, 0.7); border-radius: 20px; padding: 20px;">', unsafe_allow_html=True)
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
    st.markdown('<div style="background: rgba(20, 25, 40, 0.7); border-radius: 20px; padding: 20px;">', unsafe_allow_html=True)
    st.header(_("Predictions & AI Models", "التوقعات والنماذج الذكية"))
    
    if not data['pred2024'].empty:
        st.subheader(_("2024 Predictions", "توقعات 2024"))
        df_pred = data['pred2024']
        st.dataframe(df_pred.head(50), use_container_width=True)
    
    if not data['models'].empty:
        st.subheader(_("Model Performance", "أداء النماذج"))
        df_models = data['models']
        st.dataframe(df_models, use_container_width=True)
    
    if not data['test'].empty:
        st.subheader(_("Test Predictions", "توقعات الاختبار"))
        df_test = data['test']
        st.dataframe(df_test.head(50), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div style="background: rgba(20, 25, 40, 0.7); border-radius: 20px; padding: 20px;">', unsafe_allow_html=True)
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
<div style="background: rgba(45, 55, 72, 0.9); border-radius: 20px; padding: 30px; margin-top: 40px; text-align: center;">
    <p style="color: white; font-size: 1.2rem;">{_('Fortune 500 Analytics Dashboard', 'لوحة تحليل Fortune 500')}</p>
    <p style="color: #A0AEC0;">{_('Developed by: Mohammad Naser', 'تم التطوير بواسطة: محمد زكريا ناصر')}</p>
    <p style="color: #718096;">© 2024 {_('All Rights Reserved', 'جميع الحقوق محفوظة')}</p>
</div>
""", unsafe_allow_html=True)
