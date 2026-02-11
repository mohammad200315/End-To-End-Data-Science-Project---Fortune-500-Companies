
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import requests
import io



# ====================== Load Data ======================
try:
    df = pd.read_csv('fortune500_cleaned.csv')
    print(f"Data loaded successfully: {len(df)} rows")
except FileNotFoundError:
    print("CSV file not found! Creating sample data...")
    # Create sample data for testing
    np.random.seed(42)
    years = list(range(1996, 2024))
    companies = [f'Company {i}' for i in range(1, 501)]
    industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Energy',
                  'Manufacturing', 'Automotive', 'Telecom', 'Pharmaceuticals']

    data = []
    for year in years:
        np.random.shuffle(companies)
        for i, company in enumerate(companies[:500]):
            base_revenue = np.random.uniform(1000, 500000)
            growth = 1 + 0.05 * (year - 1996)  # 5% annual growth
            revenue = base_revenue * growth * np.random.uniform(0.8, 1.2)

            data.append({
                'year': year,
                'rank': i + 1,
                'name': company,
                'revenue_mil': revenue,
                'profit_mil': revenue * np.random.uniform(0.01, 0.2),
                'industry': np.random.choice(industries),
                'employees': int(np.random.uniform(1000, 500000)),
                'headquarters_state': np.random.choice(['CA', 'NY', 'TX', 'IL', 'FL', 'PA', 'OH', 'GA', 'NC', 'MI'])
            })

    df = pd.DataFrame(data)
    print(f"Sample data created: {len(df)} rows")

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
    'info': '#3B82F6',
    'text_light': '#FFFFFF',
    'text_dark': '#1F2937',
    'card_bg': 'rgba(255, 255, 255, 0.95)',
    'tab_bg': 'rgba(94, 58, 138, 0.1)',
}

# ====================== Main Functions ======================

def create_comprehensive_dashboard(year):
    """Create a comprehensive analysis dashboard for a specific year"""
    filtered_df = df[df['year'] == year].copy()

    if filtered_df.empty:
        return [gr.HTML(f"<div style='padding:20px;color:{COLOR_PALETTE['danger']}'><h3>No data available for year {year}</h3></div>")] * 9

    # 1. Top 10 companies
    top_10 = filtered_df.nlargest(10, 'revenue_mil')
    fig1 = px.bar(top_10, x='revenue_mil', y='name', orientation='h',
                  title=f'Top 10 Companies in {year}',
                  color='revenue_mil',
                  color_continuous_scale=[[0, COLOR_PALETTE['light']], [1, COLOR_PALETTE['primary']]],
                  labels={'revenue_mil': 'Revenue (Million $)', 'name': 'Company'},
                  hover_data=['rank', 'profit_mil', 'industry'])
    fig1.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", color=COLOR_PALETTE['text_dark'], size=12),
        title_font=dict(size=18, color=COLOR_PALETTE['primary'])
    )

    # 2. Revenue distribution
    fig2 = px.histogram(filtered_df, x='revenue_mil', nbins=50,
                        title=f'Revenue Distribution in {year}',
                        labels={'revenue_mil': 'Revenue (Million $)'},
                        color_discrete_sequence=[COLOR_PALETTE['secondary']])
    fig2.add_vline(x=filtered_df['revenue_mil'].mean(), line_dash="dash",
                   line_color=COLOR_PALETTE['accent3'],
                   annotation_text=f"Average: ${filtered_df['revenue_mil'].mean():,.0f}M")
    fig2.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')

    # 3. Industry analysis
    industry_stats = filtered_df.groupby('industry').agg({
        'revenue_mil': ['count', 'mean', 'sum'],
        'profit_margin': 'mean'
    }).round(2)
    industry_stats.columns = ['Company Count', 'Average Revenue', 'Total Revenue', 'Average Profit Margin']
    industry_stats = industry_stats.sort_values('Total Revenue', ascending=False).head(10)

    fig3 = make_subplots(rows=1, cols=2,
                         subplot_titles=('Top Industries by Revenue', 'Profit Margin by Industry'))
    fig3.add_trace(
        go.Bar(x=industry_stats.index, y=industry_stats['Total Revenue'],
               name='Total Revenue', marker_color=COLOR_PALETTE['secondary']),
        row=1, col=1
    )
    fig3.add_trace(
        go.Bar(x=industry_stats.index, y=industry_stats['Average Profit Margin'],
               name='Average Profit Margin', marker_color=COLOR_PALETTE['accent1']),
        row=1, col=2
    )
    fig3.update_xaxes(tickangle=45)
    fig3.update_layout(height=400, showlegend=False, plot_bgcolor='white', paper_bgcolor='white')

    # 4. Geographic analysis
    if 'headquarters_state' in filtered_df.columns:
        state_analysis = filtered_df.groupby('headquarters_state').agg({
            'revenue_mil': 'sum',
            'name': 'count'
        }).sort_values('revenue_mil', ascending=False).head(15)

        fig4 = px.bar(state_analysis.reset_index(),
                      x='revenue_mil', y='headquarters_state',
                      orientation='h',
                      title=f'Top 15 States by Revenue - {year}',
                      labels={'revenue_mil': 'Total Revenue (Million $)', 'headquarters_state': 'State'},
                      color='revenue_mil',
                      color_continuous_scale=[[0, COLOR_PALETTE['light']], [1, COLOR_PALETTE['accent2']]])
        fig4.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')
    else:
        fig4 = go.Figure()
        fig4.add_annotation(text="Geographic data not available", x=0.5, y=0.5, showarrow=False)
        fig4.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')

    # 5. Profit vs Revenue relationship
    fig5 = px.scatter(filtered_df, x='revenue_mil', y='profit_mil',
                      size='employees', color='industry',
                      hover_name='name',
                      title=f'Profit vs Revenue - {year}',
                      labels={'revenue_mil': 'Revenue (Million $)', 'profit_mil': 'Profit (Million $)'})
    fig5.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')

    # 6. Historical trend
    if year > df['year'].min():
        years_data = df[df['year'] <= year]
        yearly_trend = years_data.groupby('year').agg({
            'revenue_mil': 'mean',
            'profit_mil': 'mean'
        }).reset_index()

        fig6 = make_subplots(specs=[[{"secondary_y": True}]])
        fig6.add_trace(
            go.Scatter(x=yearly_trend['year'], y=yearly_trend['revenue_mil'],
                      mode='lines+markers', name='Average Revenue',
                      line=dict(color=COLOR_PALETTE['primary'], width=3)),
            secondary_y=False
        )
        fig6.add_trace(
            go.Scatter(x=yearly_trend['year'], y=yearly_trend['profit_mil'],
                      mode='lines', name='Average Profit',
                      line=dict(color=COLOR_PALETTE['accent1'], width=2, dash='dash')),
            secondary_y=True
        )
        fig6.update_layout(
            title=f'Revenue and Profit Trend (1996-{year})',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    else:
        fig6 = go.Figure()
        fig6.add_annotation(text="No historical data", x=0.5, y=0.5, showarrow=False)
        fig6.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')

    # 7. Key statistics
    stats = {
        'total_companies': len(filtered_df),
        'total_revenue': f"${filtered_df['revenue_mil'].sum():,.0f}M",
        'avg_revenue': f"${filtered_df['revenue_mil'].mean():,.0f}M",
        'avg_profit': f"${filtered_df['profit_mil'].mean():,.0f}M",
        'avg_profit_margin': f"{filtered_df['profit_margin'].mean():.1f}%",
        'top_industry': filtered_df['industry'].mode()[0] if not filtered_df['industry'].mode().empty else 'N/A'
    }

    summary_html = f"""
    <div style="background: linear-gradient(135deg, {COLOR_PALETTE['primary']} 0%, {COLOR_PALETTE['secondary']} 100%);
                padding: 25px; border-radius: 15px; color: white; margin-bottom: 20px;">
        <h2 style="margin-top: 0;">{year} Summary</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                <h4 style="margin: 0 0 10px 0;">Number of Companies</h4>
                <p style="font-size: 24px; margin: 0; font-weight: bold;">{stats['total_companies']:,}</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                <h4 style="margin: 0 0 10px 0;">Total Revenue</h4>
                <p style="font-size: 24px; margin: 0; font-weight: bold;">{stats['total_revenue']}</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                <h4 style="margin: 0 0 10px 0;">Average Profit Margin</h4>
                <p style="font-size: 24px; margin: 0; font-weight: bold;">{stats['avg_profit_margin']}</p>
            </div>
        </div>
    </div>
    """

    # 8. Top 15 companies table - UPDATED FOR BLACK TEXT
    top_companies_table = filtered_df.nlargest(15, 'revenue_mil')[
        ['rank', 'name', 'revenue_mil', 'profit_mil', 'profit_margin', 'industry']
    ].round(2)

    # Format the numbers properly
    def format_number(x):
        if isinstance(x, (int, float)):
            if x > 1000:
                return f'${x:,.0f}M'
            else:
                return f'{x:.1f}%'
        return x

    # Create HTML table with custom styling
    table_html = """
    <style>
    .company-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        color: #000000 !important;
    }
    .company-table th {
        background-color: #f8f9fa;
        color: #000000 !important;
        font-weight: bold;
        padding: 12px 15px;
        text-align: left;
        border-bottom: 2px solid #dee2e6;
    }
    .company-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #dee2e6;
        color: #000000 !important;
    }
    .company-table tr:hover {
        background-color: #f8f9fa;
    }
    </style>
    <table class="company-table">
        <thead>
            <tr>
                <th>Rank</th>
                <th>Company Name</th>
                <th>Revenue (Million $)</th>
                <th>Profit (Million $)</th>
                <th>Profit Margin</th>
                <th>Industry</th>
            </tr>
        </thead>
        <tbody>
    """

    for _, row in top_companies_table.iterrows():
        table_html += f"""
            <tr>
                <td style="color: #000000 !important;">{int(row['rank'])}</td>
                <td style="color: #000000 !important;">{row['name']}</td>
                <td style="color: #000000 !important;">${row['revenue_mil']:,.0f}M</td>
                <td style="color: #000000 !important;">${row['profit_mil']:,.0f}M</td>
                <td style="color: #000000 !important;">{row['profit_margin']:.1f}%</td>
                <td style="color: #000000 !important;">{row['industry']}</td>
            </tr>
        """

    table_html += """
        </tbody>
    </table>
    """

    final_table_html = f"""
    <div style="background: white; padding: 20px; border-radius: 15px; margin-top: 20px;">
        <h3 style="color: {COLOR_PALETTE['primary']}; margin-bottom: 15px;">Top 15 Companies in {year}</h3>
        <div style="overflow-x: auto;">
            {table_html}
        </div>
    </div>
    """

    # 9. Key insights
    largest_company = filtered_df.loc[filtered_df['revenue_mil'].idxmax()]
    profitable_industries = filtered_df.groupby('industry')['profit_margin'].mean().nlargest(3)

    insights_html = f"""
    <div style="background: white; padding: 20px; border-radius: 15px; margin-top: 20px;">
        <h3 style="color: {COLOR_PALETTE['primary']};">Key Insights</h3>
        <p style="color: #000000 !important;"><strong>Largest Company:</strong> {largest_company['name']} (${largest_company['revenue_mil']:,.0f}M)</p>
        <p style="color: #000000 !important;"><strong>Most Profitable Industries:</strong></p>
        <ul style="color: #000000 !important;">
    """
    for industry, margin in profitable_industries.items():
        insights_html += f"<li style='color: #000000 !important;'>{industry}: {margin:.1f}%</li>"
    insights_html += "</ul></div>"

    return [
        gr.HTML(summary_html),
        gr.Plot(fig1),
        gr.Plot(fig2),
        gr.Plot(fig3),
        gr.Plot(fig4),
        gr.Plot(fig5),
        gr.Plot(fig6),
        gr.HTML(final_table_html),
        gr.HTML(insights_html)
    ]

def create_company_analysis(company_name):
    """Analyze a specific company"""
    company_data = df[df['name'] == company_name].sort_values('year')

    if company_data.empty:
        return [gr.HTML(f"<div style='padding:20px;color:red'><h3>Company '{company_name}' not found</h3></div>")] * 3

    # 1. Revenue trend
    fig1 = px.line(company_data, x='year', y='revenue_mil',
                   title=f'{company_name} - Revenue Trend',
                   markers=True,
                   labels={'revenue_mil': 'Revenue (Million $)', 'year': 'Year'},
                   color_discrete_sequence=[COLOR_PALETTE['primary']])
    fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white')

    # 2. Rank trend
    fig2 = px.line(company_data, x='year', y='rank',
                   title=f'{company_name} - Rank Trend',
                   markers=True,
                   labels={'rank': 'Rank', 'year': 'Year'},
                   color_discrete_sequence=[COLOR_PALETTE['accent1']])
    fig2.update_yaxes(autorange="reversed")
    fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white')

    # 3. Statistics
    latest = company_data.iloc[-1]
    stats_html = f"""
    <div style="background: linear-gradient(135deg, {COLOR_PALETTE['accent2']} 0%, {COLOR_PALETTE['secondary']} 100%);
                padding: 25px; border-radius: 15px; color: white; margin-bottom: 20px;">
        <h2>{company_name} Profile</h2>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                <h4>Latest Revenue</h4>
                <p style="font-size: 24px; margin: 0; font-weight: bold;">${latest['revenue_mil']:,.0f}M</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                <h4>Latest Rank</h4>
                <p style="font-size: 24px; margin: 0; font-weight: bold;">#{int(latest['rank'])}</p>
            </div>
        </div>
    </div>
    """

    return [
        gr.HTML(stats_html),
        gr.Row([gr.Column([gr.Plot(fig1)]), gr.Column([gr.Plot(fig2)])]),
        gr.HTML(f"<div style='padding:20px;background:white;border-radius:10px'><h3>Historical Data</h3>{company_data[['year','rank','revenue_mil','profit_mil']].to_html(index=False)}</div>")
    ]

def create_comparison_dashboard(year1, year2):
    """Compare two years"""
    df1 = df[df['year'] == year1]
    df2 = df[df['year'] == year2]

    # Calculate growth
    total_rev_growth = ((df2['revenue_mil'].sum() - df1['revenue_mil'].sum()) / df1['revenue_mil'].sum()) * 100

    # Create chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Total Revenue', 'Average Revenue'],
                         y=[df1['revenue_mil'].sum(), df1['revenue_mil'].mean()],
                         name=str(year1),
                         marker_color=COLOR_PALETTE['primary']))
    fig.add_trace(go.Bar(x=['Total Revenue', 'Average Revenue'],
                         y=[df2['revenue_mil'].sum(), df2['revenue_mil'].mean()],
                         name=str(year2),
                         marker_color=COLOR_PALETTE['secondary']))

    fig.update_layout(
        title=f'Comparison {year1} vs {year2}',
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    growth_color = COLOR_PALETTE['success'] if total_rev_growth > 0 else COLOR_PALETTE['danger']

    summary_html = f"""
    <div style="background: linear-gradient(135deg, #E0F2FE 0%, #F0F9FF 100%);
                padding: 25px; border-radius: 15px; margin-bottom: 20px;">
        <h2 style="color: {COLOR_PALETTE['primary']};">Comparison {year1} vs {year2}</h2>
        <p style="font-size: 24px; color: {growth_color}; font-weight: bold;">
            Revenue Growth: {total_rev_growth:+.1f}%
        </p>
    </div>
    """

    return [gr.HTML(summary_html), gr.Plot(fig)]

def create_overall_insights():
    """Comprehensive insights"""
    total_years = df['year'].nunique()
    total_companies = df['name'].nunique()
    total_revenue = df['revenue_mil'].sum()

    insights_html = f"""
    <div style="background: linear-gradient(135deg, {COLOR_PALETTE['primary']} 0%, {COLOR_PALETTE['accent2']} 100%);
                padding: 30px; border-radius: 20px; color: white; margin-bottom: 20px;">
        <h2>Data Overview (1996-2023)</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 12px;">
                <h4>Number of Years</h4>
                <p style="font-size: 32px; margin: 0; font-weight: bold;">{total_years}</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 12px;">
                <h4>Unique Companies</h4>
                <p style="font-size: 32px; margin: 0; font-weight: bold;">{total_companies:,}</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 12px;">
                <h4>Total Revenue</h4>
                <p style="font-size: 32px; margin: 0; font-weight: bold;">${total_revenue/1000000:,.1f} Trillion</p>
            </div>
        </div>
    </div>
    """

    return insights_html

# ====================== Gradio Interface ======================

custom_css = """
body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; min-height: 100vh; padding: 20px; }
.gr-block { background: rgba(255, 255, 255, 0.95) !important; border-radius: 15px !important; padding: 25px !important; margin-bottom: 20px !important; }
.gr-button { background: #5E3A8A !important; color: white !important; border: none !important; padding: 12px 24px !important; border-radius: 8px !important; }
.gr-tabs { background: transparent !important; }
.gr-tab { background: rgba(94, 58, 138, 0.1) !important; border-radius: 8px !important; margin: 5px !important; padding: 15px !important; color: white !important; }
.gr-tab.selected { background: rgba(255, 255, 255, 0.25) !important; }
"""

with gr.Blocks(title="Fortune 500 Analysis Dashboard", css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    <div style="background: linear-gradient(135deg, rgba(94, 58, 138, 0.9) 0%, rgba(59, 130, 246, 0.9) 100%);
                padding: 40px; border-radius: 20px; color: white; margin-bottom: 30px; text-align: center;">
        <h1 style="margin: 0; font-size: 2.5rem;">Fortune 500 Analysis Dashboard</h1>
        <p style="margin: 15px 0 0 0; opacity: 0.9;">Comprehensive analysis of Fortune 500 companies (1996-2023)</p>
    </div>
    """)

    with gr.Tabs():
        # Tab 1: Year Analysis
        with gr.TabItem("Year Analysis"):
            year_input = gr.Dropdown(
                choices=sorted(df['year'].unique(), reverse=True),
                value=2023,
                label="Select Year"
            )

            # Outputs
            outputs = [
                gr.HTML(label="Summary"),
                gr.Plot(label="Top Companies"),
                gr.Plot(label="Revenue Distribution"),
                gr.Plot(label="Industry Analysis"),
                gr.Plot(label="Geographic Distribution"),
                gr.Plot(label="Profit vs Revenue"),
                gr.Plot(label="Historical Trend"),
                gr.HTML(label="Companies Table"),
                gr.HTML(label="Insights")
            ]

            year_input.change(create_comprehensive_dashboard, inputs=year_input, outputs=outputs)

            # Initial load
            demo.load(fn=lambda: create_comprehensive_dashboard(2023), outputs=outputs)

        # Tab 2: Company Analysis
        with gr.TabItem("Company Analysis"):
            company_input = gr.Dropdown(
                choices=sorted(df['name'].unique()),
                value="Company 1",
                label="Select Company"
            )

            company_outputs = [
                gr.HTML(label="Company Profile"),
                gr.Row([gr.Column([], scale=1), gr.Column([], scale=1)]),
                gr.HTML(label="Historical Data")
            ]

            company_input.change(create_company_analysis, inputs=company_input, outputs=company_outputs)

        # Tab 3: Comparison
        with gr.TabItem("Year Comparison"):
            with gr.Row():
                year1 = gr.Dropdown(
                    choices=sorted(df['year'].unique(), reverse=True),
                    value=2020,
                    label="First Year"
                )
                year2 = gr.Dropdown(
                    choices=sorted(df['year'].unique(), reverse=True),
                    value=2023,
                    label="Second Year"
                )

            compare_outputs = [
                gr.HTML(label="Comparison Summary"),
                gr.Plot(label="Chart")
            ]

            for input in [year1, year2]:
                input.change(create_comparison_dashboard, inputs=[year1, year2], outputs=compare_outputs)

        # Tab 4: Overall Insights
        with gr.TabItem("Overall Insights"):
            insights_output = gr.HTML()
            insights_output.value = create_overall_insights()

# ====================== Run Application ======================
if __name__ == "__main__":
    # Run on available port
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Trying another port...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False
        )
