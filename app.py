import pandas as pd
from dash import Dash, dcc, html
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

gif_path = "/Users/alya/code/AtinDhawan/DOG_IDENTIFIER/Assets/loading.gif"

app.layout = html.Div([
    html.Div([
        html.Img(src=dash.get_asset_url('Logo.png'),
                 style={'height': '40%', 'width': '40%'})
    ], style={'textAlign': 'center'}),
    html.H4("Who's that doggie in the window?",
            style={'text-align': 'center', 'font-family': 'Gill Sans', 'font-size': '32px', 'color': '#F79D59'}),
    html.P(" "),
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            dbc.Container(
                dbc.Card(
                    [
                        dbc.CardHeader("File Upload", style={'background-color': '#54BA9F', 'color': 'white',
                                                             'font-family': 'Verdana'}),
                        dbc.CardBody(
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                                style={
                                    'font-family': 'Trebuchet MS',
                                    'color': 'white',
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'solid',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '2px',
                                    'background-color': '#BDE6D3',
                                    'font-family': 'Verdana'
                                },
                                multiple=True,
                                className="upload-box",
                            )
                        ),
                    ],

                    className="theme-card",
                ),
                className="container",
                style={'margin-bottom': '20px', 'font-family': 'Verdana'}
            ),
            html.Div(
                [
                    dbc.Button('Find the breed!', color='primary', id='process-file-btn', n_clicks=0)
                ],
                style={'display': 'flex', 'justify-content': 'center', 'margin-bottom': '20px',
                       'font-family': 'Verdana'}
            ),
            dcc.Loading(
                id='loading-animation',
                type='default',
                children=[
                    html.Div(id='loading-placeholder',
                             children=[
                             html.Img(src="/Users/alya/code/AtinDhawan/DOG_IDENTIFIER/Assets/loading.gif",
                                      style={'height':'100px', 'width':'100px'})
                             ],
                            style={'textAlign': 'center', 'margin-top': '20px'})
                ]
            ),
            html.Div(id='output-data')
        ]
    )
])

def show_loading_animation(n_clicks, contents):
    if n_clicks > 0 and contents is not None:
        return {'display': 'block'}, None
    else:
        return {'display': 'none'}, None

def parse_contents(contents, filename):
    return html.Div([
        html.H5(filename),
        html.P(contents),
        html.Hr()
    ])

@app.callback(
    [Output('output-data', 'children'),
     Output('loading-animation', 'style'),
     Output('loading-placeholder', 'children'),
     Output('loading-placeholder', 'style')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')],
    [Input('process-file-btn', 'n_clicks')],
    [State('upload-data', 'contents')],
    prevent_initial_callback=True
)

def show_loading_animation(contents, filename, last_modified, n_clicks):
    loading_style = {'display': 'none'}
    loading_children = None
    output_children = None

    if n_clicks > 0 and contents is not None:
        loading_style = {'display': 'block'}
        loading_children = html.Div(
            children=[
                html.Img(
                    src=dash.app.get_asset_url('loading.gif'),
                    style={'height': '100px', 'width': '100px'}
                )
            ],
            style={'textAlign': 'center', 'margin-top': '20px'}
        )
    else:
        output_children = parse_contents(contents, filename)

    return output_children, {'display': 'block'}, loading_children, loading_style

def update_output(contents, filename, last_modified, n_clicks, contents_state):
    loading_style = {'display': 'none'}
    loading_placeholder = None

    if n_clicks > 0 and contents_state is not None:
        loading_style = {'display': 'block'}

    if n_clicks > 0:
        if contents is not None:
            children = [
                parse_contents(c, n) for c, n
                in zip(contents, filename)
            ]
            return children, loading_style, loading_placeholder

    return None, loading_style, loading_placeholder

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    