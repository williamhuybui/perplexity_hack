import plotly.express as px
import pandas as pd

def plot(final_df):
    # Make sure relevant columns are numeric
    numeric_cols = [
        'evaluation_factual_correctness_score', 
        'evaluation_completeness_score', 
        'evaluation_clarity_score'
    ]

    df = final_df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convert invalid to NaN

    # Calculate overall average score per question
    df['overall_score'] = df[numeric_cols].mean(axis=1)

    # Melt DataFrame to long format (including overall score)
    melted_df = df.melt(
        id_vars=['question'],
        value_vars=numeric_cols + ['overall_score'],
        var_name='Metric',
        value_name='Score'
    )

    # Clean up metric names for display
    melted_df['Metric'] = (
        melted_df['Metric']
        .str.replace('evaluation_', '', regex=False)
        .str.replace('_score', '', regex=False)
        .str.replace('_', ' ')
        .str.title()
    )

    # Drop rows with missing scores (optional, if needed)
    melted_df = melted_df.dropna(subset=['Score'])

    # Plot boxplot
    fig = px.box(
        melted_df,
        x='Metric',
        y='Score',
        points='all',  # show individual points
        hover_data=['question'],
        title='Score Distributions per Metric (with Overall Score)',
        height=500
    )

    fig.update_layout(
        yaxis=dict(range=[0, 6], dtick=1),
        xaxis_title='Metric',
        yaxis_title='Score (1-5)'
    )

    fig.show()