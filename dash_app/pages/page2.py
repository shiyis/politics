
import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, Output, callback
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from jax import random
from scipy import sparse
import os
import numpy as np
import dash_bootstrap_components as dbc
import matplotlib
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go


matplotlib.use('agg')

states = pd.read_csv("./data/states.csv")
candidates = pd.read_csv("./data/2022/processed_weball.csv")    


dash.register_page(__name__, title='Text-based Ideal Points',location='sidebar')


PAGE_STYLE = {
    "position":"absolute",
    "margin":"4.5rem 4rem 0rem 24.5rem",
    "color":"#000",
    "text-shadow":"#000 0 0",
    'whiteSpace': 'pre-wrap'
}


num_topics = 50
rng_seed = random.PRNGKey(0)

custom = '../tbip/data/custom_data/clean/author_map.txt' 
if not os.path.isfile(custom):
    dataPath = '../tbip/data/candidate-tweets-2020/clean/'
else:
    dataPath = '/'.join(custom.split('/')[:-1])
# Load data
author_indices = jax.device_put(jnp.load(dataPath + "author_indices.npy"), jax.devices("cpu")[0])

counts = sparse.load_npz(dataPath + "counts.npz")

with open(dataPath + "vocabulary.txt",'r') as f:
    vocabulary = f.readlines()

with open(dataPath + "author_map.txt",'r') as f:
    author_map = f.readlines()

author_map = np.array(author_map)
num_authors = int(author_indices.max() + 1)
num_documents, num_words = counts.shape
pre_initialize_parameters = True

neutral_topic_mean = np.load("../output/candidate-tweets-2020/neutral_topic_mean.npy")
negative_topic_mean = np.load("../output/candidate-tweets-2020/negative_topic_mean.npy")
positive_topic_mean = np.load("../output/candidate-tweets-2020/positive_topic_mean.npy")
authors = pd.read_csv("../output/candidate-tweets-2020/authors.csv")
authors["name"] = authors["name"].str.replace("\n", "")



layout = html.Div([html.H5("The Text-based Ideal Points Model"),
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
                         html.H5("Ideological Topics and Ideal Points Generated for Author's Political Leaning"),
                         html.Hr(),
                         dcc.Markdown("""Below are trained results for the list of 2022 federal election candidates' ideal points and topic aggregation of their Twitter archive 
                                """),
                         dbc.Row(
                                [dbc.Col(dbc.Row(dbc.Col(children=[dcc.Dropdown(pd.DataFrame(pd.read_csv("./data/states.csv"))['name'].tolist(),id='state-dropdown-p2')],id='states-col-p2'))), 
                                 dbc.Col(dbc.Row(dbc.Col(children=[dcc.Dropdown(id='names-dropdown-p2'), dcc.Checklist(id='select-all',
                                            options=['Select All'], value=[])],id='cand-names-col-p2'),id='cand-names-row-p2')),
                                 
                                 ]),

                          dbc.Row([
                                    dbc.Col([ html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                        dcc.Graph(id='bar-graph-plotly', figure={})
                                    ], width=12, md=6),
                                    dbc.Col([
                                        dag.AgGrid(
                                            id='grid',
                                            rowData=authors.to_dict("records"),
                                            columnDefs=[{"field": i} for i in authors.columns],
                                            columnSize="sizeToFit",
                                        )
                                    ], width=12, md=6),
                                ], className='mt-4'),
                         html.Br(),
                         html.Br(),
                         html.Br()     
                        ],className='page2',style=PAGE_STYLE)

@callback(
    Output('cand-names-col-p2', 'children'),
    [dash.dependencies.Input('state-dropdown-p2', 'value')])
def update_output(value):
    a = states.loc[states['name'] == value, 'state']

    dropdown = '' 
    if value:
        res = candidates.loc[candidates['State'] == a.iloc[0], 'Candidate name'].tolist()
        dropdown = [dcc.Dropdown(['Select all'] + res,id='names-dropdown-p2', value=res[0], searchable=True,  multi=True)]
        
    else:
        res = candidates['Candidate name'].tolist()
        dropdown = dcc.Dropdown(res ,id='names-dropdown-p2', value=res[0], searchable=True,  multi=True)
    return dropdown






@callback(
    Output('bar-graph-plotly', 'figure'),
    [
        dash.dependencies.Input('bar-graph-plotly', 'figure')
        # whatever other inputs
    ]
)
def my_callback(figure_empty):

    authors['split'] = authors[authors['ideal_point'] > 0]
    fig = px.scatter(authors, x=['ideal_point'], y=[1]*len(authors),  height=200,hover_data=['name'])
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    x = np.array(authors['split'].tolist())
    y = np.array([])

    fig = go.Figure()
    color = np.random.randint(2, size=len(x))


    trace1 = np.where(color==0)
    trace2 = np.where(color==1)

    fig.add_trace(go.Scatter(x=x[trace1],
                            y=y[trace1],
                            mode='markers',
                            name='false',
                            marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=x[trace2],
                            y=y[trace2],
                            mode='markers',
                            name='true',
                            marker=dict(color='green')
                            ))        
    fig.update_yaxes(
                range=(0.5, 1.5),
                constrain='domain'
        )
    return fig



@callback(
    Output(component_id='graph-matplotlib', component_property='src'),
    [
        dash.dependencies.Input('state-dropdown-p2', 'value'), dash.dependencies.Input('names-dropdown-p2', 'value')
        # whatever other inputs
    ]
)
def my_callback(state_choice, pillar_dropdown):
    a = states.loc[states['name'] == state_choice, 'state']

    selected_authors = []
    if pillar_dropdown:
        if set(pillar_dropdown) == 'Select all':
            selected_authors = candidates.loc[candidates['State'] == a.iloc[0], 'Candidate name'].tolist()
        else:
            selected_authors = pillar_dropdown

    return pillar_dropdown