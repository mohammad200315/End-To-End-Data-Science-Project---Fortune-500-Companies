import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ====================== Page Configuration ======================
st.set_page_config(
    page_title="Fortune 500 Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تطبيق CSS مخصص
st.markdown("""
<style>
    /* خلفية التطبيق */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* بطاقات */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* الأزرار */
    .stButton > button {
        background: linear-gradient(135deg, #5E3A8A 0%, #3B82F6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* التبويبات */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
        padding: 10px 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.25);
        color: white;
    }
    
    /* القوائم المنسدلة */
    .stSelectbox, .stDropdown {
        background: white;
        border-radius: 8px;
    }
    
    /* النصوص */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    .stMarkdown {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================== Load Data ======================
@st.cache_data
def load_all_data():
    """تحميل جميع ملفات البيانات"""
    data_files = {}
    
    try:
        # الملف الأساسي
        df = pd.read_csv('fortune500_cleaned.csv')
        data_files['main'] = df
        st.sidebar.success(f"✅ البيانات الرئيسية: {len(df):,} صف")
    except FileNotFoundError:
        st.sidebar.error("❌ ملف fortune500_cleaned.csv غير موجود!")
        data_files['main'] = pd.DataFrame()
    
    try:
        # توقعات 2024
        predictions_2024 = pd.read_csv('fortune500_2024_predictions.csv')
        data_files['predictions_2024'] = predictions_2024
        st.sidebar.success(f"✅ توقعات 2024: {len(predictions_2024):,} صف")
    except FileNotFoundError:
        data_files['predictions_2024'] = pd.DataFrame()
    
    try:
        # أداء النماذج
        models_performance = pd.read_csv('fortune500_models_performance.csv')
        data_files['models_performance'] = models_performance
        st.sidebar.success(f"✅ أداء النماذج: {len(models_performance)} نماذج")
    except FileNotFoundError:
        data_files['models_performance'] = pd.DataFrame()
    
    try:
        # توقعات الاختبار
        test_predictions = pd.read_csv('fortune500_test_predictions.csv')
        data_files['test_predictions'] = test_predictions
        st.sidebar.success(f"✅ توقعات الاختبار: {len(test_predictions):,} صف")
    except FileNotFoundError:
        data_files['test_predictions'] = pd.DataFrame()
    
    return data_files

# تحميل البيانات
data_files = load_all_data()
df = data_files.get('main', pd.DataFrame())

# التحقق من البيانات
if df.empty:
    st.error("""
    ⚠️ **لا توجد بيانات للعرض!**
    
    **يرجى التأكد من:**
    1. وجود ملف `fortune500_cleaned.csv` في نفس المجلد
    2. صحة اسم الملف
    3. أن الملف يحتوي على البيانات
    """)
    st.stop()

# ====================== Data Processing ======================
df['profit_margin'] = (df['profit_mil'] / df['revenue_mil']) * 100
df['revenue_per_employee'] = df['revenue_mil'] / df['employees']

# ====================== Color Palette ======================
COLOR_PALETTE = {
    'primary': '#5E3A8A',
    'secondary': '#3B82F6',
    'accent1': '#10B981',
    'accent2': '#8B5CF6',
    'accent3': '#F59E0B',
    'dark': '#1F2937',
    'light': '#F3F4F6',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'info': '#3B82F6'
}

# ====================== Header ======================
st.markdown(f"""
<div style="background: linear-gradient(135deg, {COLOR_PALETTE['primary']} 0%, {COLOR_PALETTE['secondary']} 100%);
            padding: 40px; 
            border-radius: 20px; 
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-align: center;">
    <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
        📊 Fortune 500 Analytics Dashboard
    </h1>
    <p style="margin: 15px 0 0 0; opacity: 0.9; font-size: 1.2rem;">
        تحليل شامل لبيانات شركات Fortune 500 من 1996 إلى 2023
    </p>
</div>
""", unsafe_allow_html=True)

# ====================== Sidebar ======================
with st.sidebar:
    st.markdown(f"""
    <div style="background: rgba(255, 255, 255, 0.1); 
                padding: 20px; 
                border-radius: 15px;
                margin-bottom: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);">
        <h3 style="color: white; margin-top: 0;">⚙️ لوحة التحكم</h3>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_type = st.radio(
        "**اختر نوع التحليل:**",
        ["📅 تحليل السنوات", "🏢 تحليل الشركات", "📈 مقارنة السنوات", 
         "🔮 التوقعات والتنبؤات", "📊 نظرة عامة"],
        index=0
    )

# ====================== Main Content ======================
if analysis_type == "📅 تحليل السنوات":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("📊 تحليل البيانات السنوية")
    
    # اختيار السنة
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_year = st.selectbox(
            "**اختر سنة للتحليل:**",
            sorted(df['year'].unique(), reverse=True),
            index=0
        )
    
    with col2:
        show_top = st.number_input("**عدد الشركات:**", min_value=5, max_value=50, value=15)
    
    # تصفية البيانات
    filtered_df = df[df['year'] == selected_year].copy()
    
    if filtered_df.empty:
        st.warning(f"⚠️ لا توجد بيانات للسنة {selected_year}")
    else:
        # الإحصائيات الأساسية
        st.subheader(f"📈 إحصائيات سنة {selected_year}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("**عدد الشركات**", f"{len(filtered_df):,}", 
                     help="إجمالي عدد الشركات في القائمة لهذه السنة")
        
        with col2:
            total_rev = filtered_df['revenue_mil'].sum()
            st.metric("**إجمالي الإيرادات**", f"${total_rev:,.0f}M",
                     help="مجموع إيرادات جميع الشركات بالمليون دولار")
        
        with col3:
            avg_rev = filtered_df['revenue_mil'].mean()
            st.metric("**متوسط الإيرادات**", f"${avg_rev:,.0f}M",
                     help="متوسط إيرادات الشركات بالمليون دولار")
        
        with col4:
            avg_margin = filtered_df['profit_margin'].mean()
            st.metric("**متوسط هامش الربح**", f"{avg_margin:.1f}%",
                     help="متوسط نسبة الربح إلى الإيرادات")
        
        # تبويبات التحليل
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 أفضل الشركات", "📊 توزيع الإيرادات", "🏭 تحليل الصناعات", "🗺️ التوزيع الجغرافي"])
        
        with tab1:
            st.subheader(f"🏆 أفضل {show_top} شركة في {selected_year}")
            top_companies = filtered_df.nlargest(show_top, 'revenue_mil')
            
            fig1 = px.bar(top_companies, x='revenue_mil', y='name', orientation='h',
                         title=f'أفضل {show_top} شركة حسب الإيرادات',
                         color='revenue_mil',
                         color_continuous_scale=[[0, '#E0E7FF'], [1, COLOR_PALETTE['primary']]],
                         labels={'revenue_mil': 'الإيرادات (مليون $)', 'name': 'اسم الشركة'},
                         hover_data=['rank', 'profit_mil', 'profit_margin', 'industry'])
            
            fig1.update_layout(
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial", color=COLOR_PALETTE['dark'], size=12),
                title_font=dict(size=18, color=COLOR_PALETTE['primary'])
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # جدول تفصيلي
            st.subheader("📋 جدول تفصيلي")
            display_cols = ['rank', 'name', 'revenue_mil', 'profit_mil', 'profit_margin', 'industry']
            if 'headquarters_state' in top_companies.columns:
                display_cols.append('headquarters_state')
            
            styled_df = top_companies[display_cols].copy()
            styled_df['revenue_mil'] = styled_df['revenue_mil'].apply(lambda x: f"${x:,.0f}M")
            styled_df['profit_mil'] = styled_df['profit_mil'].apply(lambda x: f"${x:,.0f}M")
            styled_df['profit_margin'] = styled_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.subheader("📊 توزيع الإيرادات")
            
            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.histogram(filtered_df, x='revenue_mil', nbins=50,
                                   title='توزيع الإيرادات',
                                   labels={'revenue_mil': 'الإيرادات (مليون $)'},
                                   color_discrete_sequence=[COLOR_PALETTE['secondary']])
                fig2.add_vline(x=filtered_df['revenue_mil'].mean(), line_dash="dash",
                             line_color=COLOR_PALETTE['accent3'],
                             annotation_text=f"المتوسط: ${filtered_df['revenue_mil'].mean():,.0f}M")
                fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig3 = px.box(filtered_df, y='revenue_mil',
                             title='مخطط الصندوق للإيرادات',
                             labels={'revenue_mil': 'الإيرادات (مليون $)'})
                fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400)
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            st.subheader("🏭 تحليل الصناعات")
            
            industry_stats = filtered_df.groupby('industry').agg({
                'revenue_mil': ['count', 'mean', 'sum'],
                'profit_margin': 'mean'
            }).round(2)
            
            industry_stats.columns = ['عدد الشركات', 'متوسط الإيرادات', 'إجمالي الإيرادات', 'متوسط هامش الربح']
            industry_stats = industry_stats.sort_values('إجمالي الإيرادات', ascending=False).head(15)
            
            col1, col2 = st.columns(2)
            with col1:
                fig4 = px.bar(industry_stats.reset_index(), x='إجمالي الإيرادات', y='industry',
                             orientation='h', title='أفضل الصناعات حسب الإيرادات',
                             color='إجمالي الإيرادات', color_continuous_scale='viridis')
                fig4.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=500)
                st.plotly_chart(fig4, use_container_width=True)
            
            with col2:
                fig5 = px.bar(industry_stats.reset_index(), x='industry', y='متوسط هامش الربح',
                             title='هامش الربح حسب الصناعة',
                             color='متوسط هامش الربح', color_continuous_scale='tealrose')
                fig5.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=500, xaxis_tickangle=45)
                st.plotly_chart(fig5, use_container_width=True)
        
        with tab4:
            if 'headquarters_state' in filtered_df.columns:
                st.subheader("🗺️ التوزيع الجغرافي")
                
                state_analysis = filtered_df.groupby('headquarters_state').agg({
                    'revenue_mil': 'sum',
                    'name': 'count'
                }).sort_values('revenue_mil', ascending=False).head(20)
                
                state_analysis.columns = ['إجمالي الإيرادات', 'عدد الشركات']
                
                fig6 = px.bar(state_analysis.reset_index(), x='إجمالي الإيرادات', y='headquarters_state',
                             orientation='h', title='أفضل الولايات حسب الإيرادات',
                             color='إجمالي الإيرادات', color_continuous_scale='sunset')
                fig6.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=500)
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("ℹ️ بيانات المقر الرئيسي غير متوفرة في هذه البيانات")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif analysis_type == "🏢 تحليل الشركات":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("🏢 تحليل أداء الشركات")
    
    # اختيار الشركة
    company_name = st.selectbox(
        "**اختر شركة:**",
        sorted(df['name'].unique()),
        index=0
    )
    
    # الحصول على بيانات الشركة
    company_data = df[df['name'] == company_name].sort_values('year')
    
    if company_data.empty:
        st.warning(f"⚠️ لا توجد بيانات للشركة {company_name}")
    else:
        # الإحصائيات الرئيسية
        latest_year = company_data.iloc[-1]
        
        st.subheader(f"📋 ملف الشركة: {company_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("**عدد السنوات في القائمة**", len(company_data))
        
        with col2:
            st.metric("**آخر إيرادات**", f"${latest_year['revenue_mil']:,.0f}M")
        
        with col3:
            st.metric("**آخر ترتيب**", f"#{int(latest_year['rank'])}")
        
        with col4:
            st.metric("**آخر هامش ربح**", f"{latest_year['profit_margin']:.1f}%")
        
        # الرسوم البيانية
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 اتجاه الإيرادات")
            fig1 = px.line(company_data, x='year', y='revenue_mil',
                          title='الإيرادات عبر السنوات',
                          markers=True,
                          labels={'year': 'السنة', 'revenue_mil': 'الإيرادات (مليون $)'})
            fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("🏆 اتجاه الترتيب")
            fig2 = px.line(company_data, x='year', y='rank',
                          title='الترتيب عبر السنوات',
                          markers=True,
                          labels={'year': 'السنة', 'rank': 'الترتيب'})
            fig2.update_yaxes(autorange="reversed")
            fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # بيانات إضافية
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 اتجاه الأرباح")
            fig3 = px.line(company_data, x='year', y='profit_mil',
                          title='الأرباح عبر السنوات',
                          markers=True,
                          labels={'year': 'السنة', 'profit_mil': 'الأرباح (مليون $)'})
            fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("📊 اتجاه هامش الربح")
            fig4 = px.line(company_data, x='year', y='profit_margin',
                          title='هامش الربح عبر السنوات',
                          markers=True,
                          labels={'year': 'السنة', 'profit_margin': 'هامش الربح (%)'})
            fig4.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        # جدول البيانات التاريخية
        st.subheader("📅 البيانات التاريخية")
        
        display_data = company_data[['year', 'rank', 'revenue_mil', 'profit_mil', 'profit_margin']].copy()
        display_data['revenue_mil'] = display_data['revenue_mil'].apply(lambda x: f"${x:,.0f}M")
        display_data['profit_mil'] = display_data['profit_mil'].apply(lambda x: f"${x:,.0f}M")
        display_data['profit_margin'] = display_data['profit_margin'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif analysis_type == "📈 مقارنة السنوات":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("📈 مقارنة بين سنتين")
    
    col1, col2 = st.columns(2)
    with col1:
        year1 = st.selectbox("**السنة الأولى:**", sorted(df['year'].unique(), reverse=True), index=3)
    
    with col2:
        year2 = st.selectbox("**السنة الثانية:**", sorted(df['year'].unique(), reverse=True), index=0)
    
    if year1 == year2:
        st.warning("⚠️ يرجى اختيار سنتين مختلفتين للمقارنة")
    else:
        # الحصول على بيانات السنوات
        df_year1 = df[df['year'] == year1]
        df_year2 = df[df['year'] == year2]
        
        # حساب النمو
        total_rev_growth = ((df_year2['revenue_mil'].sum() - df_year1['revenue_mil'].sum()) / df_year1['revenue_mil'].sum()) * 100
        avg_rev_growth = ((df_year2['revenue_mil'].mean() - df_year1['revenue_mil'].mean()) / df_year1['revenue_mil'].mean()) * 100
        
        # عرض النتائج
        st.subheader(f"📊 نتائج المقارنة: {year1} vs {year2}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            growth_color = COLOR_PALETTE['success'] if total_rev_growth > 0 else COLOR_PALETTE['danger']
            st.metric("**نمو الإيرادات الإجمالية**", f"{total_rev_growth:+.1f}%",
                     delta=f"{total_rev_growth:+.1f}%", delta_color="normal" if total_rev_growth > 0 else "inverse")
        
        with col2:
            avg_growth_color = COLOR_PALETTE['success'] if avg_rev_growth > 0 else COLOR_PALETTE['danger']
            st.metric("**نمو متوسط الإيرادات**", f"{avg_rev_growth:+.1f}%",
                     delta=f"{avg_rev_growth:+.1f}%", delta_color="normal" if avg_rev_growth > 0 else "inverse")
        
        with col3:
            company_growth = len(df_year2) - len(df_year1)
            st.metric("**تغير عدد الشركات**", f"{company_growth:+d}",
                     delta=f"{company_growth:+d}", delta_color="normal" if company_growth > 0 else "inverse")
        
        # رسم بياني للمقارنة
        comparison_data = pd.DataFrame({
            'السنة': [str(year1), str(year2)],
            'إجمالي الإيرادات': [df_year1['revenue_mil'].sum(), df_year2['revenue_mil'].sum()],
            'متوسط الإيرادات': [df_year1['revenue_mil'].mean(), df_year2['revenue_mil'].mean()],
            'عدد الشركات': [len(df_year1), len(df_year2)]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='إجمالي الإيرادات',
            x=comparison_data['السنة'],
            y=comparison_data['إجمالي الإيرادات'],
            marker_color=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']],
            text=comparison_data['إجمالي الإيرادات'].apply(lambda x: f"${x:,.0f}M"),
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='متوسط الإيرادات',
            x=comparison_data['السنة'],
            y=comparison_data['متوسط الإيرادات'],
            marker_color=[COLOR_PALETTE['accent1'], COLOR_PALETTE['accent2']],
            text=comparison_data['متوسط الإيرادات'].apply(lambda x: f"${x:,.0f}M"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='مقارنة بين السنوات',
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # جدول المقارنة التفصيلي
        st.subheader("📋 جدول المقارنة التفصيلي")
        
        comparison_table = pd.DataFrame({
            'المعيار': ['عدد الشركات', 'إجمالي الإيرادات', 'متوسط الإيرادات', 'متوسط الأرباح', 'متوسط هامش الربح'],
            f'{year1}': [
                f"{len(df_year1):,}",
                f"${df_year1['revenue_mil'].sum():,.0f}M",
                f"${df_year1['revenue_mil'].mean():,.0f}M",
                f"${df_year1['profit_mil'].mean():,.0f}M",
                f"{df_year1['profit_margin'].mean():.1f}%"
            ],
            f'{year2}': [
                f"{len(df_year2):,}",
                f"${df_year2['revenue_mil'].sum():,.0f}M",
                f"${df_year2['revenue_mil'].mean():,.0f}M",
                f"${df_year2['profit_mil'].mean():,.0f}M",
                f"{df_year2['profit_margin'].mean():.1f}%"
            ],
            'النمو': [
                f"{(len(df_year2) - len(df_year1)) / len(df_year1) * 100:+.1f}%",
                f"{total_rev_growth:+.1f}%",
                f"{avg_rev_growth:+.1f}%",
                f"{((df_year2['profit_mil'].mean() - df_year1['profit_mil'].mean()) / df_year1['profit_mil'].mean() * 100):+.1f}%",
                f"{((df_year2['profit_margin'].mean() - df_year1['profit_margin'].mean()) / df_year1['profit_margin'].mean() * 100):+.1f}%"
            ]
        })
        
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif analysis_type == "🔮 التوقعات والتنبؤات":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("🔮 التوقعات والتنبؤات")
    
    # التحقق من وجود ملفات التوقعات
    if data_files['predictions_2024'].empty:
        st.warning("""
        ⚠️ **ملفات التوقعات غير متوفرة**
        
        **الملفات المطلوبة:**
        1. `fortune500_2024_predictions.csv` - توقعات 2024
        2. `fortune500_models_performance.csv` - أداء النماذج
        3. `fortune500_test_predictions.csv` - توقعات الاختبار
        """)
    else:
        tab1, tab2, tab3 = st.tabs(["📅 توقعات 2024", "🤖 أداء النماذج", "🧪 توقعات الاختبار"])
        
        with tab1:
            st.subheader("📅 توقعات عام 2024")
            predictions_2024 = data_files['predictions_2024']
            
            if not predictions_2024.empty:
                st.success(f"✅ تم تحميل {len(predictions_2024):,} توقعات لعام 2024")
                
                # عرض أفضل 20 توقع
                st.subheader("🏆 أفضل 20 شركة متوقعة لعام 2024")
                top_predictions = predictions_2024.head(20)
                
                fig = px.bar(top_predictions, x='predicted_revenue_mil', y='name', orientation='h',
                           title='أفضل الشركات المتوقعة لعام 2024',
                           color='predicted_revenue_mil',
                           color_continuous_scale='viridis',
                           labels={'predicted_revenue_mil': 'الإيرادات المتوقعة (مليون $)', 'name': 'اسم الشركة'})
                
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # جدول التوقعات
                st.subheader("📋 جدول التوقعات الكامل")
                
                display_cols = []
                if 'name' in predictions_2024.columns:
                    display_cols.append('name')
                if 'predicted_revenue_mil' in predictions_2024.columns:
                    display_cols.append('predicted_revenue_mil')
                if 'predicted_rank' in predictions_2024.columns:
                    display_cols.append('predicted_rank')
                if 'growth_percentage' in predictions_2024.columns:
                    display_cols.append('growth_percentage')
                
                if display_cols:
                    styled_predictions = predictions_2024[display_cols].copy()
                    
                    if 'predicted_revenue_mil' in styled_predictions.columns:
                        styled_predictions['predicted_revenue_mil'] = styled_predictions['predicted_revenue_mil'].apply(
                            lambda x: f"${x:,.0f}M" if pd.notnull(x) else "N/A"
                        )
                    
                    if 'growth_percentage' in styled_predictions.columns:
                        styled_predictions['growth_percentage'] = styled_predictions['growth_percentage'].apply(
                            lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A"
                        )
                    
                    st.dataframe(styled_predictions, use_container_width=True, hide_index=True)
        
        with tab2:
            st.subheader("🤖 أداء نماذج التنبؤ")
            models_performance = data_files['models_performance']
            
            if not models_performance.empty:
                st.success(f"✅ تم تحميل بيانات {len(models_performance)} نموذج")
                
                # رسم بياني لأداء النماذج
                if 'model_name' in models_performance.columns and 'accuracy' in models_performance.columns:
                    fig = px.bar(models_performance, x='model_name', y='accuracy',
                               title='دقة النماذج التنبؤية',
                               color='accuracy',
                               color_continuous_scale='rdylgn',
                               labels={'model_name': 'اسم النموذج', 'accuracy': 'الدقة (%)'})
                    
                    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # عرض جدول النماذج
                st.dataframe(models_performance, use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("🧪 نتائج توقعات الاختبار")
            test_predictions = data_files['test_predictions']
            
            if not test_predictions.empty:
                st.success(f"✅ تم تحميل {len(test_predictions):,} توقع اختبار")
                
                # إذا كان هناك بيانات فعلية ومتوقعة
                if all(col in test_predictions.columns for col in ['actual_revenue_mil', 'predicted_revenue_mil']):
                    # رسم بياني للمقارنة
                    fig = px.scatter(test_predictions.head(100), x='actual_revenue_mil', y='predicted_revenue_mil',
                                   title='المقارنة بين القيم الفعلية والمتوقعة',
                                   trendline='ols',
                                   labels={'actual_revenue_mil': 'القيمة الفعلية (مليون $)', 
                                           'predicted_revenue_mil': 'القيمة المتوقعة (مليون $)'})
                    
                    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # عرض عينة من البيانات
                st.subheader("📋 عينة من توقعات الاختبار")
                st.dataframe(test_predictions.head(20), use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:  # نظرة عامة
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("📊 نظرة عامة على البيانات")
    
    # الإحصائيات الرئيسية
    total_years = df['year'].nunique()
    total_companies = df['name'].nunique()
    total_revenue = df['revenue_mil'].sum()
    avg_revenue_growth = df.groupby('year')['revenue_mil'].mean().pct_change().mean() * 100
    
    st.subheader("📈 نظرة عامة شاملة")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("**عدد السنوات**", total_years)
    
    with col2:
        st.metric("**عدد الشركات الفريدة**", f"{total_companies:,}")
    
    with col3:
        st.metric("**إجمالي الإيرادات**", f"${total_revenue/1000000:,.1f}T")
    
    with col4:
        st.metric("**متوسط النمو السنوي**", f"{avg_revenue_growth:.1f}%")
    
    # تبويبات التحليل العام
    tab1, tab2, tab3, tab4 = st.tabs(["📅 التطور التاريخي", "🏆 أفضل الشركات", "🏭 تحليل الصناعات", "📊 إحصائيات إضافية"])
    
    with tab1:
        st.subheader("📅 التطور التاريخي للإيرادات")
        
        # متوسط الإيرادات السنوي
        yearly_avg = df.groupby('year').agg({
            'revenue_mil': 'mean',
            'profit_mil': 'mean',
            'profit_margin': 'mean'
        }).reset_index()
        
        fig = make_subplots(rows=3, cols=1, subplot_titles=('متوسط الإيرادات السنوية', 'متوسط الأرباح السنوية', 'متوسط هامش الربح السنوي'))
        
        fig.add_trace(
            go.Scatter(x=yearly_avg['year'], y=yearly_avg['revenue_mil'], mode='lines+markers',
                      name='متوسط الإيرادات', line=dict(color=COLOR_PALETTE['primary'], width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_avg['year'], y=yearly_avg['profit_mil'], mode='lines+markers',
                      name='متوسط الأرباح', line=dict(color=COLOR_PALETTE['accent1'], width=2)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_avg['year'], y=yearly_avg['profit_margin'], mode='lines+markers',
                      name='هامش الربح', line=dict(color=COLOR_PALETTE['accent2'], width=2, dash='dot')),
            row=3, col=1
        )
        
        fig.update_layout(height=800, plot_bgcolor='white', paper_bgcolor='white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🏆 أفضل الشركات على الإطلاق")
        
        # أفضل الشركات حسب أعلى إيرادات سجلتها
        top_companies_alltime = df.groupby('name').agg({
            'revenue_mil': 'max',
            'profit_mil': 'max',
            'year': 'count'
        }).nlargest(15, 'revenue_mil')
        
        top_companies_alltime.columns = ['أعلى إيرادات', 'أعلى أرباح', 'عدد السنوات في القائمة']
        
        fig = px.bar(top_companies_alltime.reset_index(), x='أعلى إيرادات', y='name', orientation='h',
                   title='أفضل 15 شركة حسب أعلى إيرادات سجلتها',
                   color='أعلى إيرادات',
                   color_continuous_scale='sunsetdark',
                   labels={'name': 'اسم الشركة', 'أعلى إيرادات': 'الإيرادات (مليون $)'})
        
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # جدول تفصيلي
        st.subheader("📋 جدول تفصيلي لأفضل الشركات")
        
        display_df = top_companies_alltime.reset_index().copy()
        display_df['أعلى إيرادات'] = display_df['أعلى إيرادات'].apply(lambda x: f"${x:,.0f}M")
        display_df['أعلى أرباح'] = display_df['أعلى أرباح'].apply(lambda x: f"${x:,.0f}M")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("🏭 تحليل الصناعات")
        
        industry_analysis = df.groupby('industry').agg({
            'name': 'nunique',
            'revenue_mil': ['sum', 'mean'],
            'profit_margin': 'mean'
        }).round(2)
        
        industry_analysis.columns = ['عدد الشركات', 'إجمالي الإيرادات', 'متوسط الإيرادات', 'متوسط هامش الربح']
        industry_analysis = industry_analysis.sort_values('إجمالي الإيرادات', ascending=False).head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(industry_analysis.reset_index(), x='إجمالي الإيرادات', y='industry', orientation='h',
                         title='أفضل الصناعات حسب إجمالي الإيرادات',
                         color='إجمالي الإيرادات', color_continuous_scale='temps')
            fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=500)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(industry_analysis.reset_index(), x='industry', y='متوسط هامش الربح',
                         title='هامش الربح حسب الصناعة',
                         color='متوسط هامش الربح', color_continuous_scale='balance')
            fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=500, xaxis_tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.subheader("📊 إحصائيات إضافية")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # توزيع هامش الربح
            st.subheader("📈 توزيع هامش الربح")
            fig1 = px.histogram(df, x='profit_margin', nbins=50,
                              title='توزيع هامش الربح لجميع الشركات',
                              labels={'profit_margin': 'هامش الربح (%)'})
            fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            # إحصائيات العمالة
            if 'employees' in df.columns:
                st.subheader("👥 إحصائيات العمالة")
                avg_employees = df['employees'].mean()
                total_employees = df['employees'].sum()
                
                st.metric("متوسط عدد الموظفين", f"{avg_employees:,.0f}")
                st.metric("إجمالي عدد الموظفين", f"{total_employees:,.0f}")
        
        with col2:
            # العلاقة بين الإيرادات والأرباح
            st.subheader("💰 العلاقة بين الإيرادات والأرباح")
            fig2 = px.scatter(df.sample(1000) if len(df) > 1000 else df, 
                            x='revenue_mil', y='profit_mil',
                            title='العلاقة بين الإيرادات والأرباح',
                            trendline='ols',
                            labels={'revenue_mil': 'الإيرادات (مليون $)', 'profit_mil': 'الأرباح (مليون $)'})
            fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            # معلومات عن البيانات
            st.subheader("ℹ️ معلومات عن البيانات")
            st.info(f"""
            **خصائص البيانات:**
            - الفترة الزمنية: {df['year'].min()} - {df['year'].max()}
            - إجمالي الصفوف: {len(df):,}
            - عدد الأعمدة: {len(df.columns)}
            - آخر تحديث: بيانات حتى 2023
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ====================== Footer ======================
st.markdown("""
<hr style="margin: 40px 0; border: none; height: 1px; background: rgba(255,255,255,0.2);">
<div style="text-align: center; color: rgba(255,255,255,0.7); padding: 20px; font-size: 14px;">
    <p><strong>📊 Fortune 500 Analytics Dashboard</strong></p>
    <p>تم التطوير باستخدام Streamlit و Plotly | بيانات Fortune 500 من 1996 إلى 2023</p>
    <p>جميع الحقوق محفوظة © 2024</p>
</div>
""", unsafe_allow_html=True)

# ====================== Requirements.txt لل Streamlit Cloud ======================
# أضف ملف requirements.txt مع المحتوى التالي:
"""
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
"""

# ====================== تعليمات النشر على Streamlit Cloud ======================
"""
للنشر على Streamlit Cloud:

1. احفظ هذا الكود كـ `app.py`
2. أنشئ ملف `requirements.txt` بالمحتوى أعلاه
3. تأكد من وجود جميع ملفات CSV في نفس المجلد
4. ارفع إلى GitHub
5. في Streamlit Cloud:
   Repository: your-username/repo-name
   Branch: main
   Main file path: app.py
   App URL: fortune500-dashboard
"""
