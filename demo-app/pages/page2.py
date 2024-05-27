import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, Output, callback
import pandas as pd
import numpy as np
import os
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import warnings

# Suppress regex match group warning
warnings.filterwarnings(
    "ignore",
    message="This pattern is interpreted as a regular expression, and has match groups.",
)

# Your code here


plt.switch_backend("Agg")
states = pd.read_csv("./data/states.csv")
candidates = pd.read_csv("./data/2022/processed_weball.csv")

dash.register_page(__name__, title="Measuring Subjectivity", location="sidebar")


PAGE_STYLE = {
    "position": "absolute",
    "margin": "4.5rem 4rem 0rem 20rem",
    "color": "#000",
    "text-shadow": "#000 0 0",
    "whiteSpace": "pre-wrap",
}

custom = "./data/custom_data/clean/author_map.txt"


if not os.path.isfile(custom):
    dataPath = "./data/2022/candidate-tweets-2020/clean/"
else:
    dataPath = "/".join(custom.split("/")[:-1])


authors = pd.read_csv("./data/2022/authors.csv")
authors["name"] = authors["name"].str.replace("\n", "")


layout = html.Div(
    [
        html.H5("Measuring Political Stance and Subjectivity with VAE methods"),
        html.Hr(),
        html.P(
            [
                dcc.Markdown(
                    """To continue with our project objective of measuring and investigating how a politician's internal motivation aligns with their external actions (fundraising, disbursements, and various expenditures with respect to their political acitivities and agenda; we have disclosed the most basic information in the previous section, or [Exploratory Data Analysis of Federal Election Candidacy](https://my-dash-app-ilf47zak6q-uc.a.run.app/page1), this section will present a statistical topic modeling over the authors/politicians' tweets."""
                ),
                html.A(
                    "TBIP or Text-based Ideal Point Model",
                    href="https://www.aclweb.org/anthology/2020.acl-main.475/",
                ),
                """ is an unsupervised probabilistic topic model (Keyon V., Suresh N., David B. et al.) that evaluates texts to quantify the political stances of their authors. The model does not require any text labeled with an ideology, nor does it use political parties or votes.""",
                html.P(""""""),
                dcc.Markdown(
                    """Instead, it assesses the `latent political viewpoints` of text writers and how `per-topic word choice` varies according to the author's political stance `("ideological topics")` given a corpus of political text and the author of each document."""
                ),
                """Below are the resulting ideal points,\n""",
                dbc.Row(
                    [
                        dbc.Col([], id="bar-graph-plotly"),
                        dbc.Col(
                            children=[
                                dag.AgGrid(
                                    id="grid",
                                    rowData=authors.to_dict("records"),
                                    columnDefs=[{"field": i} for i in authors.columns],
                                    columnSize="sizeToFit",
                                    style={"text-align": "center"},
                                )
                            ],
                            md=20,
                            style={"margin": "2rem 0 0 0"},
                        ),
                    ],
                    className="mt-4",
                    style={"text-align": "center"},
                ),
                html.P(""),
                html.Br(),
                html.H5("Topic Modeling with Variational Encoding"),
                html.Hr(),
                """The model performs inference using """,
                html.A(
                    "variational inference.", href="https://arxiv.org/abs/1601.00670"
                ),
                """ with """,
                html.A("reparameterization", href="https://arxiv.org/abs/1312.6114"),
                html.A(" gradients.", href="https://arxiv.org/abs/1401.4082"),
                """ What this means in plain language is that imagine you have a bunch of data, like pictures of cats. Each cat picture can be described by a set of features—things like the color of the fur, the size of the ears, and the length of the tail. Now, let's say you want to understand the hidden or latent factors that contribute to these features. The challenge is that there might be some randomness or uncertainty in these latent factors.\n\n""",
                """This is an extension of another popular algorithm the Latent Dirichlet Allocation (LDA). In the context of textual topic modeling, variational inference helps approximate the posterior distribution of latent variables, such as the distribution of topics in documents and words in topics. In variational inference, we need to specify a family of distributions from which we will choose an approximation to the true (but often intractable) posterior distribution. This family of distributions is called the "variational family." Common choices for the variational family include mean-field variational families.\n\n""",
                """In summary, variational inference, with the help of a variational family, allows us to approximate complex posterior distributions in topic modeling. It helps uncover latent topics and their distributions in a collection of documents, providing valuable insights into the underlying thematic structures.""",
                html.P(""""""),
                dcc.Markdown(
                    """Now we want to see above descriptions materialized in actual formulas. Again, because it is intractable to evaluate the posterior distribution $p(\\theta, \\beta, \\eta, x | y)$, so the posterior is estimated with a distribution $q_\\phi(\\theta, \\beta,\\eta,x)$, parameterized by $\\phi$ through minimizing the KL-Divergence between $q$ and the posterior (put simple is the distance between these two distributions), which is equivalent to maximizing the ELBO (or the Evidence Lower Bound):""",
                    mathjax=True,
                ),
                dcc.Markdown(
                    """$$\\mathbb{L}_{\\theta,\phi}(\\mathbf{x})=\mathbb{E}_{q_{\\phi}(\\mathbf{z}|\\mathbf{x})}[\\log p_{\\theta}(\\mathbf{x},\\mathbf{z})-\\log q_{\\phi}(\\mathbf{z}|\\mathbf{x})]$$""",
                    mathjax=True,
                    style={"text-align": "center"},
                ),
                dcc.Markdown(
                    """The variational family is set to be the mean-field family, meaning the latent variables factorize over documents $d$, topics $k$, and authors $$s$$:""",
                    mathjax=True,
                ),
                dbc.Row(
                    [
                        dcc.Markdown(
                            """$$q_\\phi(\\theta, \\beta, \\eta, x) = \\prod_{d,k,s} q(\\theta_d)q(\\beta_k)q(\\eta_k)q(x_s)$$""",
                            mathjax=True,
                        )
                    ],
                    style={"text-align": "center"},
                ),
                dcc.Markdown(
                    """Lognormal factors are used for the positive variables and Gaussian factors for the real variables:"""
                ),
                dbc.Row(
                    [
                        dcc.Markdown(
                            """
                            $$q(\\theta_d) = \\text{LogNormal}_K(\\mu_{\\theta_d}\\sigma^2_{\\theta_d})$$
                            $$q(\\beta_k) = \\text{LogNormal}_V(\\mu_{\\beta_k}, \\sigma^2_{\\beta_k})$$
                            $$q(\\eta_k) = \\mathcal{N}_V(\\mu_{\\eta_k}, \\sigma^2_{\\eta_k})$$
                            $$q(x_s) = \\mathcal{N}(\\mu_{x_s}, \\sigma^2_{x_s})$$""",
                            mathjax=True,
                        )
                    ],
                    style={"text-align": "center"},
                ),
                dcc.Markdown(
                    """Thus, the goal is to maximize the ELBO with respect to $$\\phi = \\{\\mu_\\theta, \\sigma_\\theta, \\mu_\\beta, \\sigma_\\beta,\\mu_\\eta, \\sigma_\\eta, \\mu_x, \\sigma_x\\}$$.\n\n""",
                    """The most important is the initializations of the variational parameters $$\\phi$$ and their respective variational distributions.""",
                    mathjax=True,
                ),
                dbc.Row(
                    [
                        dcc.Markdown(
                            """
                        `loc`: location variables $\\mu$
                        `scale`: scale variables $\\sigma$
                        $\\mu_\\eta$: `ideological_topic_loc`
                        $\\sigma_\\eta$: `ideological_topic_scale`""",
                            mathjax=True,
                        )
                    ],
                    style={"text-align": "center"},
                ),
                dcc.Markdown(
                    """The corresponding variational distribution is `ideological_topic_distribution`.  Below summarizes the above formulas in plainer language. """
                ),
                dbc.Row(
                    children=[
                        dbc.Col(
                            children=[
                                """1. Mean-Field Variational Family:""",
                                html.Code(
                                    """
                                          In mean-field variational inference, it is assumed that the posterior distribution factorizes across latent variables. This means that each latent variable is independent of the others given parameters. For LDA, these parameters might include the distribution of topics in documents and words in topics."""
                                ),
                            ]
                        ),
                        dbc.Col(
                            children=[
                                """2. Init and Optimization Process:""",
                                html.Code(
                                    """
                                          Start with some initial parameters for the variational family. Optimize these parameters to make the approximating distribution as close as possible to the true posterior distribution. In this case, it's through maximizing the evdience lower bound or minimizing the KL-Divergence between these two probability distributions."""
                                ),
                            ]
                        ),
                        dbc.Col(
                            children=[
                                """3. Projecting Latent Information:""",
                                html.Code(
                                    """
                                          The parameters of the variational family can be interpreted as the estimated distributions of topics. These distributions are used to project latent information onto the documents and words. Each document gets a distribution over topics, and each topic gets a distribution over words."""
                                ),
                            ]
                        ),
                        dbc.Col(
                            children=[
                                """4. Discovering Latent Topics:""",
                                html.Code(
                                    """
                                          The algorithm, through this variational inference process, discovers latent topics in the corpora based on how words co-occur across documents. Alternatively, it's measuring the pointwise mutual information between two probability distributions."""
                                ),
                            ]
                        ),
                    ]
                ),
                """\nThe default corpus for this Colab notebook is """,
                html.A(
                    "Senate speeches", href="https://data.stanford.edu/congress_text"
                ),
                """ from the 114th Senate session (2015-2017). The project also used the following corpora: Tweets from 2022 Democratic presidential candidates.""",
            ]
        ),
        """To replicate the whole process with my own Twitter data, I followed the steps below:""",
        dcc.Markdown(
            """
        * `counts.npz`: a `[num_documents, num_words]` [sparse CSR matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html) containing the word counts for each document.
        * `author_indices.npy`: a `[num_documents]` vector where each entry is an integer in the set `{0, 1, ..., num_authors - 1}`, indicating the author of the corresponding document in `counts.npz`.
        * `vocabulary.txt`: a `[num_words]` - length file where each line denotes the corresponding word in the vocabulary.
        * `author_map.txt`: a `[num_authors]` - length file where each line denotes the name of an author in the corpus.""",
            mathjax=True,
        ),
        dcc.Markdown(
            """Please checkout this [notebook](https://colab.research.google.com/github/pyro-ppl/numpyro/blob/5291d0627d68598cf78b8ea97c540268660925c1/notebooks/source/tbip.ipynb) for the full implementation in Python."""
        ),
        html.Br(),
    ],
    className="page2",
    style=PAGE_STYLE,
)


@callback(
    Output("cand-names-row-2", "children"),
    [dash.dependencies.Input("state-dropdown", "value")],
)
def update_output(value):
    a = states.loc[states["name"] == value, "state"]
    if value:
        res = candidates.loc[
            candidates["State"] == a.iloc[0], "Candidate name"
        ].tolist()
    else:
        res = candidates["Candidate name"].tolist()
    return html.Label(
        ["Select Candidate"],
        style={
            "font-size": "13px",
            "text-align": "left",
            "off-set": 4,
            "color": "#808080",
        },
    ), dcc.Dropdown(res, id="names-dropdown", searchable=True, multi=True)


@callback(
    Output("bar-graph-plotly", "children"),
    [dash.dependencies.Input("bar-graph-plotly", "figure")],
)
def my_callback(figure_empty):
    fig = px.scatter(
        authors, x=["ideal_point"], y=[1] * len(authors), hover_data=["name"]
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    string_1 = "(D)"
    string_2 = "(R)"
    string_3 = "(I)"

    x1 = authors[authors["name"].str.contains(rf"\b{string_1}\b")]
    x2 = authors[authors["name"].str.contains(rf"\b{string_2}\b")]
    x3 = authors[authors["name"].str.contains(rf"\b{string_3}\b")]

    mr = x2[x2["ideal_point"] < 0]
    mb = x1[x1["ideal_point"] < 0]
    pr = x2[x2["ideal_point"] >= 0]
    pb = x1[x1["ideal_point"] >= 0]

    mi = x3[x3["ideal_point"] < 0]
    pi = x3[x3["ideal_point"] >= 0]

    y = np.array([1] * len(authors))

    layout = go.Layout(
        xaxis={
            "title": "Author's Ideal Point from Moderate to Progressive",
            "visible": True,
            "showticklabels": True,
        },
        yaxis={"title": "y-label", "visible": False, "showticklabels": False},
        height=230,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#f2f2f2",  # Set the background color of the plot
        # paper_bgcolor='#f2f2f2'  # Set the background color of the entire plot area
    )

    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Scatter(
            x=np.array(mr["ideal_point"]),
            y=y,
            mode="markers",
            name="moderate repub",
            marker=dict(symbol="x", color="red"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in mr["name"].tolist()],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(mb["ideal_point"]),
            y=y,
            mode="markers",
            name="conservative",
            marker=dict(color="blue"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in mb["name"].tolist()],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(mi["ideal_point"]),
            y=y,
            mode="markers",
            name="conservative",
            marker=dict(symbol="square", color="grey"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in mi["name"].tolist()],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(pi["ideal_point"]),
            y=y,
            mode="markers",
            name="conservative",
            marker=dict(symbol="square", color="grey"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in pi["name"].tolist()],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(pr["ideal_point"]),
            y=y,
            mode="markers",
            name="conservative",
            marker=dict(symbol="x", color="red"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in pr["name"].tolist()],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(pb["ideal_point"]),
            y=y,
            mode="markers",
            name="conservative",
            marker=dict(color="blue"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in pb["name"].tolist()],
            showlegend=False,
        )
    )

    return dcc.Graph(id="bar-graph-plotly", figure=fig)


@callback(
    Output(component_id="graph-matplotlib", component_property="src"),
    [
        dash.dependencies.Input("state-dropdown-p2", "value"),
        dash.dependencies.Input("names-dropdown-p2", "value"),
        # whatever other inputs
    ],
)
def my_callback(state_choice, pillar_dropdown):
    a = states.loc[states["name"] == state_choice, "state"]

    selected_authors = []
    if pillar_dropdown:
        if set(pillar_dropdown) == "Select all":
            selected_authors = candidates.loc[
                candidates["State"] == a.iloc[0], "Candidate name"
            ].tolist()
        else:
            selected_authors = pillar_dropdown

    return pillar_dropdown


@callback(
    Output(component_id="bar-graph-matplotlib", component_property="src"),
    dash.dependencies.Input("pac-cands-topic-slider", "value"),
)
def plot_data(selected_value):
    # Build the matplotlib figure
    topics = dict.fromkeys(range(selected_value))
    for i in topics.keys():
        for j in d:
            if str(i) in j:
                if topics[i] == None:
                    topics[i] = [j]
                elif len(topics[i]) < 3:
                    topics[i].append(j)
                else:
                    pass

    # Python program to generate WordCloud

    nrows = (50 + 2) // 3
    fig, axs = plt.subplots(nrows, 3, figsize=(14, 3 + 3 * nrows))
    axs = axs.flatten()

    # iterate through the csv file
    for n, i in list(topics.items())[:selected_value]:
        comment_words = ""
        for val in i:
            # typecaste each val to string
            val = str(val)
            # split the value
            tokens = [
                "".join([j for j in i.strip() if j.isalpha() and i != ""])
                for i in val[val.index(":") + 1 :].split("\\n")
            ]
            # Converts each token into lowercase

        comment_words += " ".join(tokens) + " "

        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color="white",
            stopwords=stopwords,
            min_font_size=10,
        ).generate(comment_words)

        axs[n].set_title(f"Topic {n + 1}")
        axs[n].imshow(wordcloud, interpolation="bilinear")
        axs[n].axis("off")

    # plt.show()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib = f"data:image/png;base64,{fig_data}"

    return fig_bar_matplotlib
