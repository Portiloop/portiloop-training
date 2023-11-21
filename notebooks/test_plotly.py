from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
import time

# Assume we have a model with a method `predict` that returns a value between 0 and 1
class Model:
    def predict(self):
        return random.random()

model = Model()

# while True:
# fig = go.Figure(data=[go.Bar(
#     x=['Model Output'],
#     y=[model.predict()],
#     orientation='h',
#     marker_color='rgb(26, 118, 255)'
# )])
fig = make_subplots(rows=2, cols=1, subplot_titles=("EEG", "Model Output"))
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Data'), row=1, col=1)
fig.add_trace(go.Bar(y=["Output"], x=[0.82], name='Model Output', orientation='h'), row=2, col=1)

fig.update_layout(yaxis=dict(range=[0, 1]))

fig.show()



# Update plot with new data
# fig.update_traces(x=, selector=dict(name='Data'))
# fig.update_traces(x=["Output"], y=[)], selector=dict(name='Model Output'))