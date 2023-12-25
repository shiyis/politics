# app.py

import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State
from datetime import datetime

p = os.path.abspath(__file__)
os.chdir(os.path.dirname(p))

app = dash.Dash(__name__,use_pages=True,external_stylesheets=[dbc.themes.VAPOR])
app.config['suppress_callback_exceptions'] = True

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "height": "100%",
    "padding": "8rem 2rem 0 2rem",
    "font-size":"1rem",
    "background-color":'#37136b'
}

# the styles for the main content position it to the right of the sidebar and
# # add some padding.
CONTENT_STYLE = {
    'background-color':'#fff',
    'position':'relative',
    'height':'2500px',
    "margin-left": "20rem",
}


sidebar = html.Div(
    [   
        html.P(" ", className="display-4"),
        html.Hr(),
        html.P(
            "C4FE-TBIP", className="lead"
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(page['title'], href=page['path'], active="exact") for page in dash.page_registry.values() if page['location'] == 'sidebar'
            ],
            vertical=True,
            pills=True,
            className="nav-links"),
        # html.Hr(),
        html.H1("🗳️", className="logo"),
        # dcc.Upload(
        #     id='upload-data',
        #     children=html.Div([
        #         'Drag and Drop or ',
        #         html.A('Select Files')
        #     ]),
        #     style={
        #         'width': '100%',
        #         'height': '60px',
        #         'lineHeight': '60px',
        #         'borderWidth': '1px',
        #         'borderStyle': 'dashed',
        #         'borderRadius': '0px',
        #         'textAlign': 'center',
        #         'margin': '20px 0px 0px 0px',
        #     },
        # # Allow multiple files to be uploaded
        # multiple=True
        # ),
        html.Div(id='output-data-upload'),
        html.Hr()
    ], className='sidebar',
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content",style=CONTENT_STYLE)


# #processed uploaded file
# def parse_contents(contents, filename, date):
#     content_type, content_string = contents.split(',')

#     decoded = base64.b64decode(content_string)
#     try:
#         if 'csv' in filename:
#             # Assume that the user uploaded a CSV file
#             df = pd.read_csv(
#                 io.StringIO(decoded.decode('utf-8')))
#         elif 'xls' in filename:
#             # Assume that the user uploaded an excel file
#             df = pd.read_excel(io.BytesIO(decoded))
#     except Exception as e:
#         print(e)
#         return html.Div([
#             'There was an error processing this file.'
#         ])

#     return html.Div([
#         html.H5(filename),
#         html.H6(datetime.datetime.fromtimestamp(date)),

#         dash_table.DataTable(
#             df.to_dict('records'),
#             [{'name': i, 'id': i} for i in df.columns]
#         ),

#         html.Hr(),  # horizontal line

#         # For debugging, display the raw contents provided by the web browser
#         html.Div('Raw Content'),
#         html.Pre(contents[0:200] + '...', style={
#             'whiteSpace': 'pre-wrap',
#             'wordBreak': 'break-all'
#         })
#     ])




# @app.callback(Output('output-data-upload', 'children'),
#               Input('upload-data', 'contents'),
#               State('upload-data', 'filename'),
#               State('upload-data', 'last_modified'))
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children



content =  html.Div([
    dash.page_container
],id = "page-content")


app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

           


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)