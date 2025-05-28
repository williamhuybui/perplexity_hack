import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from llm import llm_evaluate_answer

def evaluate_answers(api_key, df, verbose = True):
    """
    Df should have the following columns:
    - question: The question to evaluate.
    - chunk: The chunk of text from the document.
    - answer: The model-generated answer to evaluate.
    """
    results = []
    for _, row in df.iterrows():
        question = row['question']
        chunk = row['chunk']
        answer = row['answer']
        
        success = False
        while not success:
            try:
                evaluation = llm_evaluate_answer(api_key, question, chunk, answer)
                success = True  # Break loop if successful
            except Exception as e:
                if verbose:
                    print(f"Retrying for question: {question} due to error: {e}")

        evaluation_data = {
            'question': question,
            'factual_correctness_score': evaluation.get('factual_correctness_score', None),
            'completeness_score': evaluation.get('completeness_score', None),
            'clarity_score': evaluation.get('clarity_score', None)
        }
        results.append(evaluation_data)

    final_df = pd.DataFrame(results).fillna(0)
    final_df["overall_score"]  = 0.5* final_df["factual_correctness_score"] + 0.3* final_df["completeness_score"] + 0.2* final_df["clarity_score"]
    return final_df


def plot(final_df):
    # Make sure relevant columns are numeric
    numeric_cols = [
        'factual_correctness_score', 
        'completeness_score', 
        'clarity_score'
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


def radar_plot(df):
    """
    Create a radar chart from the first row of *df*.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the score columns (values 0–5).
    score_cols : tuple[str]
        Column names in the order you want on the radar axes.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    # Extract scores and wrap the first value to close the loop
    # scores = df.loc[0, list(score_cols)].tolist()

    score_cols=("factual_correctness_score", "completeness_score", "clarity_score", "overall_score")
    df = df[score_cols].mean()
    scores = df[score_cols].tolist()
    scores += scores[:1]

    # Axis labels must wrap too
    labels = list(score_cols) + [score_cols[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scores,
            theta=labels,
            mode="lines+markers",   # puts the points on top
            fill="toself",
            marker=dict(size=8)
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 5], dtick=1)),
        showlegend=False,
        title="Model Evaluation Radar"
    )
    return fig

import pandas as pd

def statistics(df, highlight=True):
    # 1. compute overall score


    cols = [
        "factual_correctness_score",
        "completeness_score",
        "clarity_score",
        "overall_score",
    ]

    stats = (
        df[cols]
        .describe()
        .T.assign(median=lambda x: x["50%"])
        .loc[:, ["count", "mean", "std", "min", "median", "max"]]
        .round(2)
    )
    stats["count"] = stats["count"].astype(int)

    if not highlight:
        return stats

    # 🔸 high-contrast highlight for the mean column
    dark_blue = "#003366"
    return (
        stats.style
        .set_properties(
            subset=["mean"],
            **{
                "background-color": dark_blue,
                "color": "white",
                "font-weight": "bold",
            }
        )
        .format("{:.2f}")
    )


def overall_histogram(df, bin_size=0.25):
    """
    Plot a histogram of overall scores (0‒5).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain an 'overall_score' column.
    bin_size : float, optional
        Width of each bin; defaults to 0.25.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    fig = px.histogram(
        df,
        x="overall_score",
        nbins=int(5 / bin_size),
        title="Distribution of Overall Scores",
        labels={"overall_score": "Overall Score"},
    )

    # add mean reference line
    mean_val = df["overall_score"].mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right",
    )

    fig.update_layout(bargap=0.05, xaxis=dict(dtick=0.5))
    return fig
