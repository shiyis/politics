import dash
import pandas as pd
import dash_bootstrap_components as dbc
import dash_leaflet.express as dlx 
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_leaflet as dl
from datetime import datetime
from dash_extensions.javascript import arrow_function
from dash_extensions.javascript import assign


dash.register_page(__name__, title='Exploratory Data Analaysis',location='sidebar')


candidates = pd.read_csv("./data/2022/processed_weball.csv")    
states = pd.read_csv("./data/states.csv")


PAGE_STYLE = {
    # 'background-color':'#fff',
    "position":"relative",
    "margin":"4.5rem 4rem 0rem 22rem",
    "color":"#000",
    "text-shadow":"#000 0 0",
    'whiteSpace': 'pre-wrap'
}


def get_info(feature=None):
    header = [html.H4("PAC Financial Data by States")]
    if not feature:
        return header + [html.B("Hoover over a state")]
    return header + [html.B(feature["properties"]["name"]), html.Br(),
                    "Total Received: ${:.3f} \n\nTotal Spent: ${:.3f}".format(feature["properties"]["total_r"],feature["properties"]["total_s"])]



def create_choropleth(id='geojson1', info_id='info1'):
    classes = [0, 5000000, 10000000, 50000000, 100000000, 200000000, 300000000]
    colorscale = ['#ffffed','#fcecc4','#ffd69f','#ffbb84','#ff9b78','#ff757b','#ff468e','#ff00ac']
    style = dict(weight=1, opacity=1, color='white', dashArray='', fillOpacity=0.7)

    # Create colorbar.
    ctg = ["{}+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}+".format(classes[-1])]
    colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=500, height=10, position="bottomleft")

    # Geojson rendering logic, must be JavaScript as it is executed in clientside.
    style_handle = assign("""function(feature, context){
        const {classes, colorscale, style, colorProp} = context.hideout;  // get props from hideout
        const value = feature.properties[colorProp];  // get value the determines the color
        for (let i = 0; i < classes.length; ++i) {
            if (value > classes[i]) {
                style.fillColor = colorscale[i];  // set the fill color according to the class
            }
        }
        return style;
    }""")
    # Create geojson.
    geojson = dl.GeoJSON(url="/assets/us-states.json",  # url to geojson file
                        style=style_handle,  # how to style each polygon
                        zoomToBounds=False,  # when true, zooms to bounds when data changes (e.g. on load)
                        zoomToBoundsOnClick=False,  # when true, zooms to bounds of feature (e.g. polygon) on click
                        hoverStyle=arrow_function(dict(weight=3, color='purple', opacity=0.5, dashArray='')),  # style applied on hover
                        hideout=dict(colorscale=colorscale, classes=classes, style=style, colorProp="total_r"),
                        id=id)
    # Create info control.
    info = html.Div(children=get_info(), id=info_id, className="info",
                    style={"position": "absolute", "top": "305px", "left": "530px", "zIndex": "1000"})
    return geojson,colorbar,info

choropleth1 = create_choropleth()
choropleth2 = create_choropleth(id='geojson2',info_id='info2')


map1 = dl.Map(children=[dl.TileLayer()],style={'height': '450px','margin-top':'0rem'}, center=[39, -98], zoom=4, id='candidates-stats-marker')
map2 = dl.Map(children=[dl.TileLayer()],style={'height': '450px','margin-top':'0rem'}, center=[states[states['state']== 'DC']['latitude'].iloc[0],states[states['state']== 'DC']['longitude'].iloc[0]],zoom=7, id='candidates-individual-marker')


layout =  html.Div([
                    html.H5("C4PE Exploratory Data Analysis"),
                    html.Hr(),
                    html.P("""In socio-politics, quantified approaches and modeling techniques are applied in supporting and facilitating political analyses. Individuals, parties, committees and other political entities` come together and try to push forward campaigns in hope to receive appropriate patrionization and support for their political agenda. """),
                    html.P("""The Political Action Committees (PACs or Super PACs) amass funding resources that could benefit the elections. These type of fundings could be from other individuals, or political entities. For the sole of purpose of understanding how the processes of fund raising activities like these really work, this part of the project explores the 2021-2022 PACs financial data."""),
                    html.P("""This part of the project will first present the receipts, disbursements, and other expenditures in terms of propagating political actions in visualization format grounded in states; for example, how many different political action committees there are by US states. This part of the project will also break down all the candidates of 2022 their basic information as mentioned above including their basic demographics, political party affiliation, election cycle, and incumbency."""),
                    html.P("""All info is retrievable through the Federal Election Commission's directory. This project seeks to conduct the research with full transparency and abide to relevant conduct code."""),
                    dbc.Row(
                        [dbc.Col(dbc.Row(dbc.Col(children=[dcc.Dropdown(pd.DataFrame(pd.read_csv("./data/states.csv"))['name'].tolist(),id='state-dropdown')],id='states-col'))),
                         dbc.Col(dbc.Row(dbc.Col(children=[dcc.Dropdown(id='names-dropdown')],id='cand-names-col'),id='cand-names-row'))]),
                    # html.P(),
                    html.Div(html.Div(children=[html.Div(dcc.Slider(7000, 27500000, 3000000,value=3000000,id='pac-exp-filter'),style={'margin':'2rem -1.3rem 0rem -1.3rem'})],id='cand-names-col')),
                    html.P(),
                    html.Div(id='mapmessage', style={'color' : '#FFFFFF', 'fontSize' : '20px', 'marginTop' : '-25px'}),
                    html.Br(),
                    html.Br(),
                    dbc.Row(
                        [dbc.Col(dbc.Row(dbc.Col(children=[map1])),id='map1-col'),
                         dbc.Col(dbc.Row(dbc.Col(children=[map2])),id='map2-col')])
],className='page1',id='page1-content', style=PAGE_STYLE)

@callback(
    Output('cand-names-row', 'children'),
    [dash.dependencies.Input('state-dropdown', 'value')])
def update_output(value):
    a = states.loc[states['name'] == value, 'state']
    if value:
        res = candidates.loc[candidates['State'] == a.iloc[0], 'Candidate name'].tolist()
    else:
        res = candidates['Candidate name'].tolist()
    return dcc.Dropdown(res,id='names-dropdown',searchable=True)


@callback(Output('candidates-stats-marker', 'children'), Output('candidates-individual-marker','viewport'), Output('candidates-individual-marker', 'children'),
    [dash.dependencies.Input('pac-exp-filter', 'value'),dash.dependencies.Input('state-dropdown', 'value'), dash.dependencies.Input('names-dropdown','value')])

def update_output(slider, state, cand):
        latLon = candidates[['Party code','Party affiliation','Affiliated Committee Name','Total receipts','Total disbursements','lat','lon']]
        latLon = [tuple(i[1:]) for i in latLon.itertuples()]
        colors = ['red','blue','grey']
        s_latlon = [states[states['state']== 'DC']['latitude'].iloc[0],states[states['state']== 'DC']['longitude'].iloc[0]]

        groups = {'DEM': ('blue',[]), 'REP':('red',[]), 'OTH': ('grey',[])}

        for code,pty,name,r,s,lat,lng in latLon:
            if r >= 1000000000:
                radius = 50
                opacity = 0.8
            if r >= 20000000:
                radius = 25
                opacity = 0.8
            elif r >= 10000000:
                radius = 20
                opacity = 0.7

            elif r >= 5000000:
                radius = 18
                opacity = 0.6

            elif r  >= 4000000:
                radius = 15
                opacity = 0.5
            elif r >= 3000000:
                radius = 11
                opacity = 0.4
            elif r >= 2000000:
                radius = 9
                opacity = 0.3
            elif r >= 1000000:
                radius = 7
                opacity = 0.3
                radius = 5
                opacity = 0.3
            elif r == 0:
                radius = 2
                opacity = 0.2
            else:
                radius = 4
                opacity = 0.3


            if r > slider and r < slider + 2500000:

                cm = dl.CircleMarker(center=[lat, lng], color=colors[int(code)-1], opacity=opacity, weight=1, fillColor=colors[int(code)-1],fillOpacity=opacity, radius=radius,children=[dl.Popup(f'Committee Name: \n {name} \n Election cycle: 2022 \n Total Raised (YTD2022): {r} \n Total Spent (YTD2022): {s}')])
                if pty != 'REP' and pty != 'DEM':
                    groups['OTH'][1].append(cm)
                else:
                    groups[pty][1].append(cm)
        # point_to_layer = assign("function(feature, latlng, context) {return L.toolTip('testing!');}")
        data = [dl.TileLayer(), dl.LayersControl([dl.Pane(name='US-Boundaries',children=[dl.BaseLayer(name='US-Boundaries',children=choropleth1,checked=True)]),
                    dl.Pane(name='PAC-Data',children=[dl.Overlay(
                        dl.LayerGroup(groups['DEM'][1]), name='DEM', checked=True),
                        dl.Overlay(dl.LayerGroup(groups['REP'][1]), name='REP', checked=True),
                        dl.Overlay(dl.LayerGroup(groups['OTH'][1]), name='3RD', checked=True)])])]
        

        second_map = data.copy()
        second_map.remove(second_map[1])
        second_map.insert(1, dl.LayersControl([dl.Pane(name='US-Boundaries',children=[dl.BaseLayer(name='US-Boundaries',children=choropleth2,checked=True)]),
                    dl.Pane(name='PAC-Data',children=[dl.Overlay(
                        dl.LayerGroup(groups['DEM'][1]), name='DEM', checked=True),
                        dl.Overlay(dl.LayerGroup(groups['REP'][1]), name='REP', checked=True),
                        dl.Overlay(dl.LayerGroup(groups['OTH'][1]), name='3RD', checked=True)])]))
        if state:
            if states[states['name']==state]['capital'].iloc[0]:
                s_latlon = [states[states['name']==state]['latitude_c'].iloc[0],states[states['name']== state]['longitude_c'].iloc[0]]
            else:
                s_latlon = [states[states['name']==state]['latitude'].iloc[0],states[states['name']== state]['longitude'].iloc[0]]
            if cand:
                row = candidates[candidates['Candidate name'] == cand]
                latlng = [row['lat'].iloc[0],row['lon'].iloc[0]]   
                m = dl.Marker(position=latlng,children=[dl.Popup(f'Committee Name: \n {row["Affiliated Committee Name"].iloc[0]} \n Election cycle: 2022 \n Total Raised (YTD2022): {row["Total receipts"].iloc[0]} \n Total Spent (YTD2022): {row["Total disbursements"].iloc[0]}')])
                second_map.append(dl.Pane(name='Individual-pin',children=[m]))
                s_latlon = [row['lat'].iloc[0],row['lon'].iloc[0]] 
       

        else:
            if cand: 
                row = candidates[candidates['Candidate name'] == cand]
                latlng = [row['lat'].iloc[0],row['lon'].iloc[0]]   
                m = dl.Marker(position=latlng,children=[dl.Popup(f'Committee Name: \n {row["Affiliated Committee Name"].iloc[0]} \n Election cycle: 2022 \n Total Raised (YTD2022): {row["Total receipts"].iloc[0]} \n Total Spent (YTD2022): {row["Total disbursements"].iloc[0]}')])
                second_map.append(dl.Pane(name='Individual-pin',children=[m]))
                s_latlon = [row['lat'].iloc[0],row['lon'].iloc[0]]  
                
        return data, dict(center=s_latlon, zoom=7, transition='flyTo'), second_map

@callback(Output("info1", "children"), dash.dependencies.Input("geojson1", "hoverData"))
def info_hover(feature):
    return get_info(feature)


@callback(Output("info2", "children"), dash.dependencies.Input("geojson2", "hoverData"))
def info_hover(feature):
    return get_info(feature)