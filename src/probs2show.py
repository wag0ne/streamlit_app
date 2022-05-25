import numpy as np
import pandas as pd

import plotly.express as px
from plotly.graph_objects import Figure

CLASSES = ["Clear/Mild", "Moderate", "Severe", "Very severe"]

def plot_model_confidence(probs: np.ndarray) -> Figure:
    prob_df = pd.DataFrame(probs.T, columns=["Probabilities"], index=CLASSES)

    fig = px.bar(
        x=prob_df["Probabilities"], 
        y=prob_df.index, 
        orientation='h', 
        range_x=[0, 1],
    )
    
    fig.update_traces(
        marker_color='rgb(158,202,225)', 
        marker_line_color='rgb(8,48,107)', 
        marker_line_width=1.5, opacity=0.6,
    )

    fig.update_layout(
        title_text="Model Confidence",  
        yaxis=dict(title="Acne severity"),
        xaxis=dict(title="Probabilities"),
    )

    return fig
