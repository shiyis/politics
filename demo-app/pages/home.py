import dash
from dash import html
import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_leaflet as dl
from datetime import datetime

dash.register_page(
    __name__, path="/", title="Project Intro & Roadmap", location="sidebar"
)


PAGE_STYLE = {
    "margin": "-2rem 0rem 0rem 15rem",
    "font-family": "system-ui",
    "background-color": "#e2def2",
    "padding-top": "3rem",
}


layout = html.Div(
    [
        html.Div(
            [
                html.Img(
                    src="assets/intro.png",
                    style={
                        "width": "1100px",
                        "height": "auto",
                        "display": "block",
                        "margin": "0 auto",
                        "border-radius": "5px",
                    },
                )
            ],
        )
    ],
    className="home",
    id="home-content",
    style=PAGE_STYLE,
)
