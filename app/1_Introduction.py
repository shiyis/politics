import streamlit as st
import os
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from folium.plugins import MarkerCluster

p = os.path.dirname(os.path.abspath(__file__))
os.chdir(p)


def toc():
    st.markdown(" **$\\qquad  \\, \\,\\quad$  [1️⃣ Introduction ](#intro)**")
    st.markdown(" **$\\qquad  \\,\\, \\quad$  [2️⃣ Data Analysis ](#eda)**", unsafe_allow_html=True)
    st.markdown(" **$\\qquad  \\,\\, \\quad$  [3️⃣ The TBIP Model ](#tbip)**") 
    st.markdown(" **$\\qquad  \\,\\, \\quad$  [4️⃣ Final Results ](#analysis)**")
    
def upload_file():
    st.markdown("---")
    uploaded_file = st.file_uploader("Feed Your Own Data (CSV file)")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = uploaded_file.getvalue().decode('utf-8').splitlines()         
        st.session_state["preview"] = ''
        for i in range(0, min(5, len(data))):
            st.session_state["preview"] += data[i]

def iframe():
    st.subheader("     ", anchor="intro")
    st.markdown("""<style>
                    .st-emotion-cache-z5fcl4 {
                        padding-top: 0rem;
                        margin-top: -7.5rem;

                    }
    
                    body {
                        line-height: 1.7;
                        letter-spacing: 0em;
                    }

                    p{
                        margin: 0 0 0;
                    }
                    [data-testid="stSidebarUserContent"] {
                        background-image: url("https://raw.githubusercontent.com/shiyis/c4pe-tbip/master/app/pages/static/ballot-box-with-ballot.251x256%20(1).png");
                        background-size: 15%;
                        background-repeat: no-repeat;
                        padding-bottom: 60px;
                        background-position: 133px 160px;
                        position: relative;
                        # background-position-x: center;
                        line-height:2.7
                    }

                    .st-emotion-cache-16txtl3 {
                        padding: 10.5rem 2rem 0;
                    }
          
                    [data-testid="stVerticalBlock"]{
                        margin-top: 1.5rem;
                        display:block;
                    }
                  
                    .st-bj {
                        -webkit-box-align: start;
                        align-items: center;
                    }
            </style>""", unsafe_allow_html=True)
    components.iframe(src="https://gamma.app/embed/yjmv7s7hjm5zyau", width=None,height=2680, scrolling=False)



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




    # for i in range(0,len(pac)):
    #     folium.Marker(
    #         location=[pac.iloc[i]['lat'], pac.iloc[i]['lon']],
    #         popup=pac.iloc[i]['committee_name'],
    #     callback=callback).add_to(mc)
    # candidates = pd.read_csv("./data/2022/processed_weball22.csv")    
    latLon = candidates[['Party code','Party affiliation','Affiliated Committee Name','Total receipts','Total disbursements','lat','lon']]
    latLon = [tuple(i[1:]) for i in latLon.itertuples()]

    m = folium.Map(location=[38, -96.5], zoom_start=4)
    mc = MarkerCluster(name='PACs Geolocations').add_to(m)        

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

    
    col1, col2 = st.columns([1,1])

    # affiliated_committee = pd.read_csv("./data/2022")
    # create selection box
    with col1:
        st.selectbox('Candidates', candidates['Candidate name'].tolist())
    with col2: 
        st.selectbox('States', [i.upper() for i in states['name'].tolist()])

    st.markdown("---")

    # display map
     
    st_map = st_folium(m,height=450, use_container_width=True)
    st.markdown("""<style>
                    [title="streamlit_folium.st_folium"] {
                        height: 550px;
                    }
                    </style>
                """,   unsafe_allow_html=True)   


    return st_map

def main():    
    #Load Data
    st.subheader("Presidential Election Candidates Twitter Archive Exploratory Data Analysis", anchor="eda")
    st.markdown("$$\\quad$$")
    df_pac = pd.read_csv('./data/2022/processed_weball.csv')


    st.markdown("""
                In socio-politics, quantified approaches and modeling techniques are applied in supporting and facilitating political analyses. Individuals, parties, committees and other political entities come together and try to push forward campaigns in hope to receive appropriate patrionization and support for their political agenda. 
                The Political Action Committees (PACs or Super PACs) amass funding resources that could benefit the elections. These type of fundings could be from other individuals, or political entities. For the sole of purpose of understanding how the processes of fund raising activities like these really work, this part of the project explores the 2021-2022 PACs financial data. The data is retrievable through the FEC's database. 

                This part of the project will first present the receipts, disbursements, and other expenditures in terms of propagating political actions in visualization format grounded in states; for example, how many different political action committees there are by US states. The accumulated receipts and expenditures. 
                This part of the project will also break down all the candidates of 2022 their basic information as mentioned above including their basic demographics, political party affiliation, election cycle, and incumbency. The candidates information is retrievable through the Federal Election Commission's directory. This project seeks to conduct the research with full transparency and abide to relevant conduct code. 

                """)

    display_map(df_pac)

    st.subheader("Quantifying Political Subjectivity with Text-based Ideal Points Clustering",anchor="tbip")
    # st.markdown("---")

    st.markdown("""
        [TBIP](https://www.aclweb.org/anthology/2020.acl-main.475/) is an unsupervised probabilistic topic model called (Keyon V., Suresh N., David B. et al.) evaluates texts to quantify the political stances of their authors. The model does not require any text labeled with an ideology, nor does it use political parties or votes. The TBIP assesses the latent political viewpoints of text writers and how per-topic word choice varies according to the author's political stance ("ideological topics") given a corpus of political text and the authors of each document. For further details, see the [paper](www.aclweb.org/anthology/2020.acl-main.475/) is the URL.
        The default corpus for this Colab notebook is [Senate speeches](https://data.stanford.edu/congress_text) from the 114th Senate session (2015-2017).
        [In the paper](https://www.aclweb.org/anthology/2020.acl-main.475/), the project uses the following corpora: Senate speeches, tweets from senators, and tweets from 2022 Democratic and 2020 presidential candidates.

        

        The next cell provides the data directory. The directory in the cell below links to speeches from the 114th Senate session from the `tbip` repo.

        To use the own corpus, upload the following fthe files to the Colab working directory:

        * `counts.npz`: a `[num_documents, num_words]` [sparse CSR matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html) containing the word counts for each document.
        * `author_indices.npy`: a `[num_documents]` vector where each entry is an integer in the set `{0, 1, ..., num_authors - 1}`, indicating the author of the corresponding document in `counts.npz`.
        * `vocabulary.txt`: a `[num_words]`-length file where each line denotes the corresponding word in the vocabulary.
        * `author_map.txt`: a `[num_authors]`-length file where each line denotes the name of an author in the corpus.

        Perform Inference

        The model performs inference using [variational inference](https://arxiv.org/abs/1601.00670) with [reparameterization](https://arxiv.org/abs/1312.6114) [gradients](https://arxiv.org/abs/1401.4082). 

        Because it is intractable to evaluate the posterior distribution $p(\\theta, \\beta, \\eta, x | y)$, so the posterior is estimated with a distribution $q_\\phi(\\theta, \\beta,\\eta,x)$, parameterized by $\\phi$ through minimizing the KL-Divergence between $q$ and the posterior, which is equivalent to maximizing the ELBO:
                
        $$\\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad  \\qquad \\qquad \\qquad \\mathbb{E}_{q_\\phi}[\\log p(y, \\theta, \\beta, \\eta, x) - \\log q_{\\phi}(\\theta, \\beta, \\eta, x)].$$
                
        The variational family is set to be the mean-field family, meaning the latent variables factorize over documents $d$, topics $k$, and authors $s$:
        $$q_\\phi(\\theta, \\beta, \\eta, x) = \\prod_{d,k,s} q(\\theta_d)q(\\beta_k)q(\\eta_k)q(x_s).$$
                
        Lognormal factors are used for the positive variables and Gaussian factors for the real variables:
        
        $$ \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad q(\\theta_d) = \\text{LogNormal}_K(\\mu_{\\theta_d}\\sigma^2_{\\theta_d})$$
                
        $$ \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad q(\\theta_d) = \\text{LogNormal}_V(\\mu_{\\beta_k}, \\sigma^2_{\\beta_k})$$
                
        $$ \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad q(\\eta_k) = \\mathcal{N}_V(\\mu_{\\eta_k}, \\sigma^2_{\\eta_k})$$
                
        $$ \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad \\qquad q(x_s) = \\mathcal{N}(\\mu_{x_s}, \\sigma^2_{x_s}).$$

        Thus, the goal is to maximize the ELBO with respect to $$\\phi = \\{\\mu_\\theta, \\sigma_\\theta, \\mu_\\beta, \\sigma_\\beta,\\mu_\\eta, \\sigma_\\eta, \\mu_x, \\sigma_x\\}$$.

        In the cells below, the initializations of the variational parameters $$\\phi$$ take place and their respective variational distributions. 
            
        - `loc`: location variables $\\mu$
        - `scale`: scale variables $\\sigma$
        - $\\mu_\\eta$: `ideological_topic_loc`
        - $\\sigma_\\eta$: `ideological_topic_scale`

        The corresponding variational distribution is `ideological_topic_distribution`.

        ```
        # Create Lognormal variational family for document intensities (theta).

        document_loc = tf.get_variable(
            "document_loc",
            initializer=tf.constant(np.log(initial_document_loc)))
                    
        document_scale_logit = tf.get_variable(
            "document_scale_logit",
            shape=[num_documents, num_topics],
            initializer=tf.initializers.random_normal(mean=-2, stddev=1.),
            dtype=tf.float32)
                    
        document_scale = tf.nn.softplus(document_scale_logit)
                    
        document_distribution = tfp.distributions.LogNormal(
            loc=document_loc,
            scale=document_scale)

        # Create Lognormal variational family for objective topics (beta).
        objective_topic_loc = tf.get_variable(
            "objective_topic_loc",
            initializer=tf.constant(np.log(initial_objective_topic_loc)))
                    
        objective_topic_scale_logit = tf.get_variable(
            "objective_topic_scale_logit",
            shape=[num_topics, num_words],
            initializer=tf.initializers.random_normal(mean=-2, stddev=1.),
            dtype=tf.float32)
                    
        objective_topic_scale = tf.nn.softplus(objective_topic_scale_logit)
                    
        objective_topic_distribution = tfp.distributions.LogNormal(
            loc=objective_topic_loc,
            scale=objective_topic_scale)

        # Create Gaussian variational family for ideological topics (eta).
        ideological_topic_loc = tf.get_variable(
            "ideological_topic_loc",
            shape=[num_topics, num_words],
            dtype=tf.float32)
                    
        ideological_topic_scale_logit = tf.get_variable(
            "ideological_topic_scale_logit",
            shape=[num_topics, num_words],
            dtype=tf.float32)
        ideological_topic_scale = tf.nn.softplus(ideological_topic_scale_logit)
                    
        ideological_topic_distribution = tfp.distributions.Normal(
            loc=ideological_topic_loc,
            scale=ideological_topic_scale)

        # Create Gaussian variational family for ideal points (x).
        ideal_point_loc = tf.get_variable(
            "ideal_point_loc",
            shape=[num_authors],
            dtype=tf.float32)
                    
        ideal_point_scale_logit = tf.get_variable(
            "ideal_point_scale_logit",
            initializer=tf.initializers.random_normal(mean=0, stddev=1.),
            shape=[num_authors],
            dtype=tf.float32)
        ideal_point_scale = tf.nn.softplus(ideal_point_scale_logit)
                    
        ideal_point_distribution = tfp.distributions.Normal(
            loc=ideal_point_loc,
            scale=ideal_point_scale)

        # Approximate ELBO.
        elbo = tbip.get_elbo(counts,
                            document_indices,
                            author_indices,
                            author_weights,
                            document_distribution,
                            objective_topic_distribution,
                            ideological_topic_distribution,
                            ideal_point_distribution,
                            num_documents,
                            batch_size)
        loss = -elbo
        ```
    """)


    st.subheader("Analyze Results",anchor="analysis")
    # st.markdown('---')

    st.markdown(""" 
        
                
        Coninuing the discussion from previous section ([TBIP in code](https://c4pe2022-tbip.streamlit.app/TBIP_In_Code)), we analyzed the modeled outcomes here.
        Before that let's do a little eda first. 
                    
        Below we load the ideal points and ideological topics.
        ```

        import os
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        ideal_point_mean = np.load("ideal_point_mean.npy")
        neutral_topic_mean = np.load("neutral_topic_mean.npy")
        negative_topic_mean = np.load("negative_topic_mean.npy")
        positive_topic_mean = np.load("positive_topic_mean.npy")


        Now we load the list of authors. If you used the own corpus, change the following line to `data_dir = '.'`.


        data_dir = 'tbip/data/senate-speeches-114/clean'
        author_map = np.loadtxt(os.path.join(data_dir, 'author_map.txt'),
                                dtype=str,
                                delimiter='\\t',
                                comments='//')
        ```

        For example, here is a graph of the learned ideal points. We don't label each point because there are too many to plot. Below we select some authors to label.


        ```

        selected_authors = np.array(
            ["Bernard Sanders (I)", "Elizabeth Warren (D)", "Charles Schumer (D)",
            "Susan Collins (R)", "Marco Rubio (R)", "John Mccain (R)",
            "Ted Cruz (R)"])


        sns.set(style="whitegrid")
        fig = plt.figure(figsize=(12, 1))
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        for index in range(len(author_map)):
        if "(R)" in author_map[index] or ideal_point_mean[index] > 0:
            color = 'red'
        else:
            color = 'blue'
        ax.scatter(ideal_point_mean[index], 0, c=color, s=20)
        if author_map[index] in selected_authors:
            ax.annotate(author_map[index], xy=(ideal_point_mean[index], 0.),
                        xytext=(ideal_point_mean[index], 0), rotation=30, size=14)
        ax.set_yticks([])
        ```

        ---
        
        Now let's create ideological topics wordcloud to show the disitrbutions and latent topics extracted with respect to how these words serve
        as a function to reflect the politician's stance.
                    
        
        




        Finally Let's create the ideal point with your uploaded data




        """, unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Exploratory Data Analysis",
        page_icon="🗳",
        layout="wide"
    )
    

    with st.sidebar:
        st.write("---")
        toc()
        # st.write(" $$\\qquad $$")
        # st.write(" $$\\qquad $$")

        # st.write(" $$\\qquad $$")
        st.write("---")
        st.sidebar.radio("$$ \qquad \qquad$$ Explore Datasets", ["C4PE2020Tweets","C4PE2022Tweets","SENATESPCH114"]) 
        upload_file()
    
    
    iframe()
    main()
    
