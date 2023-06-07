import pandas as pd
from dash import Dash, dcc, html
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

app.layout = html.Div([
    html.Div([
        html.Img(src=dash.get_asset_url('Logo.png'),
                 style = {'height':'80%', 'width':'80%'})
    ], style = {'textAlign':'center'}),
    html.H4("Who's that doggie in the window?",
        style={'text-align': 'center', 'font-family':'Gill Sans', 'font-size':'32px', 'color': '#F79D59' }),
    html.P(" "),
    dbc.Container(
        dbc.Card(
            [
                dbc.CardHeader("File Upload", style={'background-color':'#54BA9F', 'color':'white', 'font-family':'Verdana'}),
                dbc.CardBody(
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                        style={
                            'font-family':'Trebuchet MS',
                            'color':'white',
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'solid',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'background-color': '#BDE6D3',
                            'font-family':'Verdana'
                        },
                        multiple=True,
                        className="upload-box",
                    )
                ),
            ],
            
            className="theme-card",
        ),
        className="container",
        style={'margin-bottom': '20px', 'font-family':'Verdana'}
    ),
    html.Div(
        [
            dbc.Button('Find the breed!', color='primary', id='process-file-btn', n_clicks=0)
        ],
        style={'display':'flex', 'justify-content':'center', 'margin-bottom': '20px', 'font-family':'Verdana'}
    ),
    html.Div(id='output-data')
])                     

def parse_contents(contents, filename):
    return html.Div([
        html.H5(filename),
        html.P(contents),
        html.Hr()
    ])

@app.callback(Output('output-data', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              Input('process-file-btn', 'n_clicks'))

def update_output(list_of_contents, list_of_names, list_of_dates, process_file_clicks):
    if process_file_clicks > 0:
        if list_of_contents is not None:
            children = [
                parse_contents(c, n) for c, n
                in zip(list_of_contents, list_of_names)
            ]
            return children

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    