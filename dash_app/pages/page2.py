
import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_leaflet as dl
from datetime import datetime



import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
neutral_topic_mean = np.load("neutral_topic_mean.npy")
negative_topic_mean = np.load("negative_topic_mean.npy")
positive_topic_mean = np.load("positive_topic_mean.npy")
authors = pd.read_csv("authors.csv")
authors["name"] = authors["name"].str.replace("\n", "")


dash.register_page(__name__, title='Text-based Ideal Points',location='sidebar')


PAGE_STYLE = {
    "position":"absolute",
    "margin":"4.5rem 4rem 0rem 24.5rem",
    "color":"#000",
    "text-shadow":"#000 0 0",
    'whiteSpace': 'pre-wrap'
}



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
                         html.H5("Ideological Topics Generated for Author's Political Leaning"),
                         html.Hr(),
                         html.P(" ")
                         ],className='page2',style=PAGE_STYLE)


@callback(Output(),Input(''))
def callback(values):
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(12, 1))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    for index in range(authors.shape[0]):
    ax.scatter(authors["ideal_point"][index], 0, c='black', s=20)
    if authors["name"][index] in selected_authors:
        ax.annotate(author_map[index],
                    xy=(authors["ideal_point"][index], 0.),
                    xytext=(authors["ideal_point"][index], 0), rotation=30, size=14)
    ax.set_yticks([])
    return 