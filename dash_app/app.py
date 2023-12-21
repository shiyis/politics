# app.py

import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_leaflet as dl
from datetime import datetime

p = os.path.abspath(__file__)
os.chdir(os.path.dirname(p))

app = dash.Dash(external_stylesheets=[dbc.themes.VAPOR])
app.config['suppress_callback_exceptions'] = True

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "height": "100%",
    "padding": "8rem 2rem 0 2rem",
    "font-size":"1rem"
}

# the styles for the main content position it to the right of the sidebar and
# # add some padding.
CONTENT_STYLE = {
    'background-color':'#fff',
    'position':'relative',
    'height':'2500px',
    "margin-left": "20rem",
}

PAGE_STYLE = {
    'background-color':'#fff',
    "position":"absolute",
    "margin":"4.5rem 6rem 0rem 6rem",
    "color":"#000",
    "text-shadow":"#000 0 0",
    'whiteSpace': 'pre-wrap'
}



sidebar = html.Div(
    [   
        html.P(" ", className="display-4"),
        html.Hr(),
        html.P(
            "C4PE-TBIP", className="lead"
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Project Intro and Roadmap", href="/", active="exact"),
                dbc.NavLink("Exploratory Data Analysis", href="/page-1", active="exact"),
                dbc.NavLink("The Text-based Ideal Points", href="/page-2", active="exact"),
                dbc.NavLink("Feed Your Own Tweets Below!", href="/page-3", active="exact"),

            ],
            vertical=True,
            pills=True,
        ),
        # html.Hr(),
        html.H1("🗳️", className="logo"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '0px',
                'textAlign': 'center',
                'margin': '20px 0px 0px 0px',
            },
        # Allow multiple files to be uploaded
        multiple=True
        ),
        html.Div(id='output-data-upload'),
        html.Hr()
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content",style=CONTENT_STYLE)


#processed uploaded file
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])




@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([
            html.Iframe(src="https://gamma.app/embed/yjmv7s7hjm5zyau",
                style={"height": "2500px", "width": "100%"}),
        ])
    elif pathname == "/page-1":
        return html.Div([html.H5("C4PE Exploratory Data Analysis"),
                         html.Hr(),
                         html.P("""In socio-politics, quantified approaches and modeling techniques are applied in supporting and facilitating political analyses. Individuals, parties, committees and other political entities` come together and try to push forward campaigns in hope to receive appropriate patrionization and support for their political agenda. """),
                         html.P("""The Political Action Committees (PACs or Super PACs) amass funding resources that could benefit the elections. These type of fundings could be from other individuals, or political entities. For the sole of purpose of understanding how the processes of fund raising activities like these really work, this part of the project explores the 2021-2022 PACs financial data."""),
                         html.P("""This part of the project will first present the receipts, disbursements, and other expenditures in terms of propagating political actions in visualization format grounded in states; for example, how many different political action committees there are by US states. This part of the project will also break down all the candidates of 2022 their basic information as mentioned above including their basic demographics, political party affiliation, election cycle, and incumbency."""),
                         html.P("""All info is retrievable through the Federal Election Commission's directory. This project seeks to conduct the research with full transparency and abide to relevant conduct code."""),
                         dbc.Row(
                             [dbc.Col(
                                 dbc.Row([dcc.Dropdown(pd.DataFrame(pd.read_csv("./data/states.csv"))['name'].tolist(),id='state-dropdown')])),
                              dbc.Col(
                                 dbc.Row([dcc.Dropdown(pd.DataFrame(pd.read_csv("./data/2022/processed_weball.csv"))['Candidate name'].tolist(), id='cands-name-dropdown')])),
                             ]),
                         html.Div(html.Div(children=[
                                  html.Div(dcc.Slider(0, 30000000, 2500000,value=10000000,id='pac-exp-filter'),style={'margin':'2rem -1.3rem 0rem -1.3rem'})])),
                         html.Div(id='mapmessage', style={'color' : '#FFFFFF', 'fontSize' : '20px', 'marginTop' : '-20px'}),
                         html.Br(),
                         html.Br(),
                         html.Iframe(id='map', srcDoc="dash_app/data/2022/pac_weball.html", width="100%",height="450",style={"padding":"1rem 0 0 0"}),
                         ],style=PAGE_STYLE)
    
    elif pathname == "/page-2":
        return html.Div([html.H5("The Text-based Ideal Points Model"),
                         html.Hr(),
                         html.P([html.A("TBIP", href="https://www.aclweb.org/anthology/2020.acl-main.475/"), """ is an unsupervised probabilistic topic model called (Keyon V., Suresh N., David B. et al.) evaluates texts to quantify the political stances of their authors. The model does not require any text labeled with an ideology, nor does it use political parties or votes. Instead, it assesses the latent political viewpoints of text writers and how per-topic word choice varies according to the author's political stance ("ideological topics") given a corpus of political text and the authors of each document. The default corpus for this Colab notebook is """,html.A("Senate speeches", href="https://data.stanford.edu/congress_text"), """ from the 114th Senate session (2015-2017). The project also used the following corpora: Tweets from 2022 Democratic presidential candidates.""",]),
                         html.P(["""To replicate the whole process with my own Twitter data, I followed the steps below:
                                """,
                                dcc.Markdown("""
                                              
                                        * `counts.npz`: a `[num_documents, num_words]` [sparse CSR matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html) containing the word counts for each document.
                                        * `author_indices.npy`: a `[num_documents]` vector where each entry is an integer in the set `{0, 1, ..., num_authors - 1}`, indicating the author of the corresponding document in `counts.npz`.
                                        * `vocabulary.txt`: a `[num_words]`-length file where each line denotes the corresponding word in the vocabulary.
                                        * `author_map.txt`: a `[num_authors]`-length file where each line denotes the name of an author in the corpus.
                                             
                                """),
                                """Perform Inference: the model performs inference using """,html.A("variational inference", href="https://arxiv.org/abs/1601.00670"), 
                                """ with """, html.A("reparameterization", href="https://arxiv.org/abs/1312.6114"),html.A(" gradients.", href="https://arxiv.org/abs/1401.4082"), html.P(""),
                                dcc.Markdown("""Because it is intractable to evaluate the posterior distribution $p(\\theta, \\beta, \\eta, x | y)$, so the posterior is estimated with a distribution $q_\\phi(\\theta, \\beta,\\eta,x)$, parameterized by $\\phi$ through minimizing the KL-Divergence between $q$ and the posterior, which is equivalent to maximizing the ELBO:
                                """,mathjax=True),
                                dcc.Markdown("""        
                                $$\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad  \\qquad \\qquad \\qquad \\mathbb{E}_{q_\\phi}[\\log p(y, \\theta, \\beta, \\eta, x) - \\log q_{\\phi}(\\theta, \\beta, \\eta, x)].$$
                                        
                                The variational family is set to be the mean-field family, meaning the latent variables factorize over documents $d$, topics $k$, and authors $$s$$:
                                             
                                $$\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad q_\\phi(\\theta, \\beta, \\eta, x) = \\prod_{d,k,s} q(\\theta_d)q(\\beta_k)q(\\eta_k)q(x_s).$$
                                        
                                Lognormal factors are used for the positive variables and Gaussian factors for the real variables:
                                
                                $$\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad q(\\theta_d) = \\text{LogNormal}_K(\\mu_{\\theta_d}\\sigma^2_{\\theta_d})$$
                                        
                                $$\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad q(\\beta_k) = \\text{LogNormal}_V(\\mu_{\\beta_k}, \\sigma^2_{\\beta_k})$$
                                        
                                $$\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad q(\\eta_k) = \\mathcal{N}_V(\\mu_{\\eta_k}, \\sigma^2_{\\eta_k})$$
                                        
                                $$\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad q(x_s) = \\mathcal{N}(\\mu_{x_s}, \\sigma^2_{x_s}).$$

                                Thus, the goal is to maximize the ELBO with respect to $$\\phi = \\{\\mu_\\theta, \\sigma_\\theta, \\mu_\\beta, \\sigma_\\beta,\\mu_\\eta, \\sigma_\\eta, \\mu_x, \\sigma_x\\}$$.

                                The most important is the initializations of the variational parameters $$\\phi$$ take place and their respective variational distributions. 
                                    
                                $\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad$ - `loc`: location variables $\\mu$
                                $\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad$ - `scale`: scale variables $\\sigma$
                                $\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad$ - $\\mu_\\eta$: `ideological_topic_loc`
                                $\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad$ - $\\sigma_\\eta$: `ideological_topic_scale`

                                The corresponding variational distribution is `ideological_topic_distribution`. 
                                Please checkout this [notebook](https://colab.research.google.com/github/pyro-ppl/numpyro/blob/5291d0627d68598cf78b8ea97c540268660925c1/notebooks/source/tbip.ipynb) for the full implementation in Python.
                                """, mathjax=True),]),
                         html.P("  "),
                         html.H5("Ideological Topics Generated for Author's Political Leaning"),
                         html.Hr()],style=PAGE_STYLE)
    else:
        return html.Div(children=[html.P("Oh cool, this is page 3!")], style=PAGE_STYLE)
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-4 bg-light rounded-4",
    )

@callback(
    Output('slider-output-container', 'children'),
    Input('pac-exp-filter', 'value'))
def update_output(value):

    return 'You have selected "{}"'.format(value)



@app.callback(dash.dependencies.Output('map', 'srcDoc'),
                [dash.dependencies.Input('State', 'value'),
                dash.dependencies.Input('Cadidate Name', 'value'),
                dash.dependencies.Input('Disbursement', 'value'),]
)
def display_map(candidates):

    # m = folium.Map(location=[38, -96.5], zoom_start=4, scrollWheelZoom=False, tiles='CartoDB positron')
    states = pd.read_csv('./data/states.csv')

    # callback = """\
    #             function (row) {
    #                 var icon, marker;
    #                 icon = L.AwesomeMarkers.icon({
    #                     icon: "map-marker", markerColor: "blue"
    #                 });
    #                 marker = L.marker(new L.LatLng(row[-2], row[-1]));
    #                 marker.setIcon(icon);
    #                 return marker;
    #             };
    #             """
    
    # choropleth = folium.Choropleth(
    #     geo_data='./data/us-state-boundaries.geojson',
    #     name="choropleth",
    #     data=pac,
    #     columns=["committee_state", "receipts"],
    #     key_on="feature.properties.name",
    #     fill_color="YlGn",
    #     line_opacity=0.2,
    #     fill_opacity=0.7,
    #     highlight=True,
    #     legend_name="Total PAC Receipts By State, YTD 2022"
    # )


    # choropleth.geojson.add_to(m)



    latLon = candidates[['Party code','Party affiliation','Affiliated Committee Name','Total receipts','Total disbursements','lat','lon']]
    latLon = [tuple(i[1:]) for i in latLon.itertuples()]

    m = folium.Map(location=[38, -96.5], zoom_start=4)
    colors = ['blue', 'red', 'grey']
    
    # point_layer name list
    all_gp = []
    for x in range(len(latLon)):
        v = latLon[x][1]
        if latLon[x][0] == int(3):
            v = '3RD'
        pg = (latLon[x][0],v)
        all_gp.append(pg)
    
    # Create point_layer object
    unique_gp = list(set(all_gp))
    vlist = []
    for i,k in enumerate(unique_gp):
        locals()[f'point_layer{i}'] = folium.FeatureGroup(name=k[1])
        vlist.append(locals()[f'point_layer{i}'])
    
    # Creating list for point_layer
    pl_group = []
    for n in all_gp:
        for v in vlist: 
            if n[1] == vars(v)['layer_name']:
                pl_group.append(v)
    
    for n, ((code,pty,name,r,s,lat,lng),pg) in enumerate(zip(latLon, pl_group)):
        if r - s > 100000000:
            radius = 40
            weight = 12
        elif r - s > 1000000:
            radius = 30
            weight = 8
        elif r - s > 100000:
            radius = 20
            weight = 6
        elif r - s > 10000:
            radius = 10
            weight = 4
        elif r - s > 5000:
            radius = 4
            weight = 2
        elif r - s > 1000:
            radius = 2
            weight = 1
        else:
            radius = 1
            weight = 1


        iframe = folium.IFrame("Committee Name: "+ str(name) + "<br>" + "Election cycle: 2022" + "<br>" + "Total Raised (YTD2022): $" + str(r) + "<br>" + "Total Spent (YTD2022): $" + str(s))
        folium.CircleMarker(location=[lat, lng], radius=radius,weight=0,
            popup=folium.Popup(iframe, min_width=320, max_width=320, min_height=50,max_height=100), 
            tooltip="Name: " + str(name) + " Lat: " + str(lat) + " , Long: " + str(lng),
            fill=True,  # Set fill to True
            color=colors[int(code)-1],
            fill_opacity=0.5,line_opacity=0.3).add_to(pg)
        pg.add_to(m)
    folium.LayerControl().add_to(m)
    m.save("../dash_app/data/2022/PAC_Weball.html")
    
    col1, col2 = st.columns([1,1])

    # affiliated_committee = pd.read_csv("./data/2022")
    # create selection box
    # display map
     
    st_map = st_folium(m,height=450, use_container_width=True)
    st.markdown("""<style>
                    [title="streamlit_folium.st_folium"] {
                        height: 550px;
                    }
                    </style>
                """,   unsafe_allow_html=True)   

    m.save("./data/2022/mypacmap.html")
    return open('./data/2022/mypacmap.html.html', 'r').read()

if __name__ == "__main__":
    app.run_server(debug=True, port=8888)