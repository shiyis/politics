
import dash
from dash import html
import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_leaflet as dl
from datetime import datetime
dash.register_page(__name__, path='/',title='Project Intro and Roadmap',location='sidebar')

PAGE_STYLE = {
    'background-color':'#fff',
    "position":"relative",
    "margin":"15rem 6rem 0rem 6rem",
    "color":"#000",
    "text-shadow":"#000 0 0",
    'whiteSpace': 'pre-wrap'
}

layout = html.Div([
            html.Iframe(src="https://gamma.app/embed/yjmv7s7hjm5zyau",
                style={"height": "2500px", "width": "80%"}),
        ], className='home')