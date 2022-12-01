import plotly.express as px
import dash_bootstrap_components as dbc
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import pandas as pd
from app import app, client
from measure_stance import stance_measure
import plotly.express as px



mentions_layout = html.Div(
    [
        dbc.Row(
            [
                # dbc.Col(
                #     [
                #         html.Label("Number of results to return"),
                #         dcc.Dropdown(
                #             id="count-mentions",
                #             multi=False,
                #             value=20,
                #             options=[
                #                 {"label": "10", "value": 10},
                #                 {"label": "20", "value": 20},
                #                 {"label": "30", "value": 30},
                #             ],
                #             clearable=False,
                #         ),
                #     ],
                #     width=3,
                # ),
                dbc.Col(
                    [
                        html.Label("Account Username"),
                        dcc.Input(
                            id="input-handle",
                            type="text",
                            placeholder="Mentioning this account",
                            value="POTUS",
                        ),
                    ],
                    width=3,
                ),
            ],
            className="mt-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Button(
                            id="hit-button",
                            children="Submit",
                            style={"background-color": "blue", "color": "white"},
                        )
                    ],
                    width=2,
                )
            ],
            className="mt-2",
        ),
        dbc.Row(
          dbc.Col(html.P(""))
        ),
        dbc.Row(
            [
                dbc.Col([html.Div(id='profile_description', style={'whiteSpace': 'pre-line'})], width=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="stance", figure={})], width=6),
            ]
        ),
    ]
)


# pull data from twitter and create the figures
@app.callback(
    # Output(component_id="myscatter2", component_property="figure"),
    Output(component_id="profile_description", component_property="children"),
    Output(component_id="stance", component_property="figure"),

    Input(component_id="hit-button", component_property="n_clicks"),
    # State(component_id="count-mentions", component_property="value"),
    State(component_id="input-handle", component_property="value"),
)
def display_value(nclicks, acnt_handle):
    user = client.get_user(
        username=acnt_handle,
        user_fields=['description']).data

    description = user.description if len(user.description) > 0 else "This user has no profile bio"

    scores = []
    for dim in stance_measure.dims:
        scores.append(stance_measure.get_dim_score(description, dim))

    fig = px.bar(x=scores, y=[d[0] for d in stance_measure.dim_poles], orientation='h',
                 color=scores, color_continuous_scale='Bluered_r', range_color=[-1, 1], width=1000, height=400)
    fig.update_xaxes(range=[-1, 1])

    return description, fig
