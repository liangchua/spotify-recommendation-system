# Libraries

import altair as alt
import base64
import matplotlib.pyplot as plt
import os
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from PIL import Image


#%% Static Path

# path of dataset
DATA_PATH = os.path.join(os.getcwd(),'spotify.csv')

# path of images
spotify1 = Image.open('spotify-logo.png')
spotify2 = Image.open('spotify-logo2.png')


#%% Functions

# change backgound by image
def add_bg(image_file):
    with open(image_file,'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: 3200px;
            }}
        </style>
        """,
        unsafe_allow_html=True
        )

# detect and count outliers
def outliers_count(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)].count()
    return outliers

# data cleaning
def cleaning(df):
    drop_null = df.dropna()
    clean_df = drop_null.drop_duplicates()
    clean_df['release_date'] = pd.to_datetime(clean_df['release_date'])
    return clean_df


# recommendation list
def similar_tracks(track_id, track_df, track_scaled, cosine_sim):
    track_idx = track_df.index[track_df['id'] == track_id][0]
    sim_scores = list(enumerate(cosine_sim[track_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(track_df.iloc[i]['name'], track_df.iloc[i]['artists'], score) 
                  for i, score in sim_scores if i != track_idx]
    return sim_scores


#%% User Interface (streamlit.apps)

# set webpage title and icon
st.set_page_config(page_title="Spotify RecSys", page_icon=spotify2)

# set background using CSS
st.markdown(
    """
    <style>
    
    /* tab's background setting */
    button[data-baseweb="tab"] {
        background-color: #383838;
        width: 100px;
        height: 30px;
        border-radius: 15px 15px 0 0;
        }
    
    /* tab's font setting */
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-weight: bold;
        }
    
    /* sidebar's font setting */
    div[class="css-k7vsyb e16nr0p31"] > h4 > em {
        font-size: 12px;
        }
    
    /* table's even columns setting */
    table td:nth-child(even) {
        background-color: #383838 !important;
        font-size: 12px !important;
        }
    
    /* table's odd columns setting */
    table td:nth-child(odd) {
        background-color: #232323 !important;
        font-size: 12px !important;
        }
    
    /* table's setting */
    table th{
        background-color: #2a4a35 !important;
        color: white !important;
        font-size: 12px !important;
        font-weight: bold !important;
        }
    
    </style>
    """,
    unsafe_allow_html=True
)

# set background by calling "add_bg" function
add_bg('bg2.jpg')

# set page header with logo
headcols = st.columns([1,4])
with headcols[0]: 
    # spotify logo
    st.image(spotify2, width=90)
with headcols[1]: 
    # title
    st.markdown("""# Spotify Song Recommender""")

# set sidebar header
with st.sidebar:
    
    sidecols = st.columns([1,5,1])
    with sidecols[1]:
        # spotify logo 
        st.image(spotify1, width=200)
    
    # project overview
    st.markdown('''
                ### Project Overview
                
                #### *In today's digital age, music streaming platform like Spotify\
                have revolutionized the way we listen to music. With millions \
                of songs available at out fingertips, it can be overwhelming to \
                find new music that align with our tastes and preferences. This \
                is where recommendation system come in - they can analyze a user's\
                listening history and suggest new songs that they are likely to \
                enjoy. The objective of this project is to build a recommendation\
                system using Spotify data that can provide accurate and relevant \
                song recommendations to users. By doing so, we aim to enhance the \
                user experience and help users discover new music that resonates\
                with them.* ####
                ''')

# add pagination
tabs = st.tabs(['Overview','Explore','Deploy'])


#%% Tab 1: Preview

with tabs[0]: 
    
    # read data from .csv file and take few data as sample
    data = pd.read_csv(DATA_PATH, encoding='unicode_escape')[:3000]
    
    st.write("##### Data Preview #####")
    # show uploaded data
    st.table(data.head())
    
    # table 1: general statistic 
    stats = {'Number of Rows': [len(data)],
             'Number of Columns': [len(data.columns)],
             'Missing Cells': [data.isnull().sum().sum()],
             'Missing Cells (%)': [round((data.isnull().sum().sum())/(len(data)*len(data.columns))*100,2)],
             'Duplicate Rows': [data.duplicated().sum()],
             'Duplicate Rows (%)': [round(data.duplicated().sum()/len(data)*100,2)]
             }
    stats_display = pd.DataFrame(data=stats).astype(str).transpose()
    stats_display.rename(columns = {0:"Values"}, inplace=True)
    
    # table 2: count total categorical and numerical columns
    fea_type = {'Categorical Columns': [len(data.select_dtypes(include=['object']).columns)],
                'Numerical Columns': [len(data.select_dtypes(exclude=['object']).columns)]
                }
    fea_type_display = pd.DataFrame(data=fea_type).astype(str).transpose()
    fea_type_display.rename(columns = {0:"Values"}, inplace=True)
    
    # display the above results in streamlit app
    st.write("##### Descriptive Statistics #####")
    stats_cols = st.columns(2)
    with stats_cols[0]:
        # table 1
        st.table(stats_display)
    with stats_cols[1]:
        # table 2
        st.table(fea_type_display)


#%% Tab 2: Explore

with tabs[1]:
    
    # separate features based on categorical and numerical data type
    num_features = ['acousticness', 'danceability', 'duration_ms', 'energy',
                    'explicit', 'instrumentalness', 'key', 'liveness', 
                    'loudness', 'mode', 'popularity', 'speechiness', 'tempo',
                    'valence', 'year']
    cat_features = ['artists', 'id', 'name', 'release_date']
    
    with st.expander('Numerical Features Details'):
        
        selected_fea = st.selectbox('Select a feature',num_features,index=9)
        
        # visualize the selected numerical features
        dataviz = pd.DataFrame(data[selected_fea].value_counts()).reset_index()
        dataviz.columns = ['data_values','count']
        chart = alt.Chart(dataviz).mark_bar().encode(
                x = 'data_values',
                y = 'count',
                color = alt.condition(
                alt.datum.count == max(dataviz['count']),
                alt.value('#1ed760'),
                alt.value('#2a4a35'),
                )
            )
        chart = chart.configure(background='#232323')
        st.altair_chart(chart, use_container_width=True)
        
        # count and display the general statistics of selected numerical feature
        num_cols = st.columns(2)
        with num_cols[0]:
            datades = pd.DataFrame(data[selected_fea].describe())
            datades.rename(columns = {selected_fea: 'values'}, inplace=True)
            st.table(datades)
        with num_cols[1]:
            stats = {'distinct values': data[selected_fea].nunique(), 
                     'missing cells': [data[selected_fea].isnull().sum().sum()],
                     'missing cells (%)': [round((data[selected_fea].isnull().sum().sum())/len(data[selected_fea])*100,2)],
                     'outliers': outliers_count(data[selected_fea]),
                     'outliers (%)': [round(outliers_count(data[selected_fea])/len(data[selected_fea])*100,2)]
                     }
            datastats = pd.DataFrame(data=stats).astype(str).transpose()
            datastats.rename(columns = {0:'values'}, inplace=True)
            st.table(datastats)
    
    with st.expander('Categorical Features Details'):
        
        selected_fea = st.selectbox('Select a feature',cat_features,index=0)
        
        # visualize the selected categorical features
        dataviz = pd.DataFrame(data[selected_fea].value_counts()).reset_index()
        dataviz.columns = ['data_values','count']
        chart = alt.Chart(dataviz).mark_bar().encode(
                x = 'data_values',
                y = 'count',
                color = alt.condition(
                alt.datum.count == max(dataviz['count']),
                alt.value('#1ed760'),
                alt.value('#2a4a35'),
                )
            )
        chart = chart.configure(background='#232323')
        st.altair_chart(chart, use_container_width=True)
        
        # count and display the general statistics of selected categorical feature
        num_cols = st.columns(2)
        with num_cols[0]:
            datades = pd.DataFrame(data[selected_fea].describe())
            datades.rename(columns = {selected_fea: 'values'}, inplace=True)
            st.table(datades)
        with num_cols[1]:
            stats = {'missing cells': [data[selected_fea].isnull().sum().sum()],
                     'missing cells (%)': [round((data[selected_fea].isnull().sum().sum())/len(data[selected_fea])*100,2)],
                     }
            datastats = pd.DataFrame(data=stats).astype(str).transpose()
            datastats.rename(columns = {0:'values'}, inplace=True)
            st.table(datastats)
    
    with st.expander('Correlation'):
        
        # get the correlation score
        corr = data.corr()
        
        # plot the heatmap
        fig, ax = plt.subplots(facecolor="#2a4a35")
        ax = sns.heatmap(corr, annot=True, cmap='viridis', annot_kws={"fontsize":5})
        
        # set the fontsize of both x and y axis
        ax.xaxis.set_tick_params(labelsize=6, labelcolor="white", color="white")
        ax.yaxis.set_tick_params(labelsize=6, labelcolor="white", color="white")
        
        # set the fontsize of colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6, labelcolor="white", color="white")
        
        # display the chart in streamlit 
        st.pyplot(fig)
    
    with st.expander('Explore'):
        
        # let users select x and y columns
        xy_cols = st.columns(2)
        with xy_cols[0]:
            x_column = st.selectbox("Select a feature ( *x-axis* )", num_features, index=0)
        with xy_cols[1]:
            y_column = st.selectbox("Select a feature ( *y-axis* )", num_features, index=1)
        
        # create scatter plot with some restriction setting 
        if "data" in locals() and not data.empty and x_column != y_column:
            fig = px.scatter(data, x=x_column, y=y_column, width=670, height=400)
            fig.update_traces(marker=dict(color='#1ed760'))
            st.plotly_chart(fig)
        elif x_column == y_column:
            st.error("Please select different columns for x and y axes.")
        else:
            st.info("Upload data and select x and y columns to view a scatter plot.")
        


#%% Tab 3: Deploy

with tabs[2]:
    
    # data cleaning by calling the "cleaning" function
    clean_data = cleaning(data)
    
    # feature selection
    fea_sel = clean_data[['acousticness','danceability','energy','tempo']]
    
    # normalization
    scaler = MinMaxScaler()
    scaler.fit(fea_sel)
    scaled_df = scaler.transform(fea_sel)
    
    # cosine similarity matrix
    cosine_sim = cosine_similarity(scaled_df)
    
    # let users select user id from list 
    id_list = data['id']
    
    id_cols = st.columns(2)
    with id_cols[0]:
        selected_id = st.selectbox('Select a User ID', id_list)
    
    # recommendation
    similar_tracks = similar_tracks(selected_id, data, scaled_df, cosine_sim)
    similar_tracks = pd.DataFrame(similar_tracks)
    similar_tracks.columns = ['name','artists','score']
    
    # display top 10 recommended songs 
    st.write(' ')
    st.markdown('##### Songs Recommended')
    st.table(similar_tracks[1:11])
    
    # display user's historical songs
    user_his = data[data['id'] == selected_id][['name', 'artists']]
    st.markdown("##### User's Historical Records")
    st.table(user_his)


