## Deep Charts Youtube Channel: https://www.youtube.com/@DeepCharts
## Subscribe for more AI/Machine Learning/Data Science Tutorials

##################################
## 1. Data Import
##################################

import os
import markdown
import pandas as pd
from fasthtml.common import *
from fastcore.basics import NotStr
import plotly.express as px
import nfl_data_py as nfl



##################################
## 2. Initialize FastHTML app
##################################

app, rt = fast_app()



##################################
## 3. Input and Process Markdown Blog Files
##################################

# Directory containing Markdown files
POSTS_DIR = 'posts'

# Load and convert Markdown files to HTML
def load_posts():
    posts = []
    # List all Markdown files with their full paths
    md_files = [os.path.join(POSTS_DIR, f) for f in os.listdir(POSTS_DIR) if f.endswith('.md')]
    # Sort files by last modified time in descending order
    md_files.sort(key=os.path.getmtime, reverse=True)
    for filepath in md_files:
        with open(filepath, 'r', encoding='utf-8') as file:
            html_content = markdown.markdown(file.read())
            title = os.path.basename(filepath).replace('_', ' ').replace('.md', '').title()
            posts.append({"title": title, "content": html_content})
    return posts



##################################
## 4. Function to import, wrangle, and graph data 
##################################

# Generate NFL Cumulative Offensive Yards Chart
def generate_offensive_yards_chart():
    # Fetch play-by-play data for the 2024 season
    df = nfl.import_pbp_data([2024])

    # Filter for rushing and passing plays
    rushing_plays = df[df['play_type'] == 'run']
    passing_plays = df[df['play_type'] == 'pass']

    # Group by offensive team and week, then sum yards gained
    weekly_rushing_yards = rushing_plays.groupby(['posteam', 'week'])['yards_gained'].sum().reset_index()
    weekly_passing_yards = passing_plays.groupby(['posteam', 'week'])['yards_gained'].sum().reset_index()

    # Add a 'play_type' column
    weekly_rushing_yards['play_type'] = 'Rushing'
    weekly_passing_yards['play_type'] = 'Passing'

    # Combine the dataframes
    combined_df = pd.concat([weekly_rushing_yards, weekly_passing_yards])

    # Pivot the table to have teams as columns and weeks as rows
    pivot_df = combined_df.pivot_table(index='week', columns=['posteam', 'play_type'], values='yards_gained', fill_value=0)

    # Calculate cumulative yards
    cumulative_yards = pivot_df.cumsum()

    # Reset index for plotting
    cumulative_yards = cumulative_yards.reset_index()
    cumulative_yards.columns = ['week'] + [f'{team}_{ptype}' for team, ptype in cumulative_yards.columns[1:]]

    # Melt the dataframe for Plotly Express
    melted_df = cumulative_yards.melt(id_vars=['week'], var_name='team_playtype', value_name='cumulative_yards')
    melted_df[['team', 'play_type']] = melted_df['team_playtype'].str.split('_', expand=True)

    # Create Plotly Express figure
    fig = px.line(melted_df, x='week', y='cumulative_yards', color='team', facet_col='play_type',
                  title='Cumulative Offensive Yards by Week (2024 Season)',
                  labels={'week': 'Week', 'cumulative_yards': 'Cumulative Yards'},
                  category_orders={'play_type': ['Rushing', 'Passing']})

    fig.update_layout(legend_title_text='Team')
    fig.update_xaxes(type='category')

    return fig.to_html(full_html=False, include_plotlyjs='cdn')



##################################
## 5. Homepage Route for Content Layout
##################################

@rt('/')
def home():
    posts = load_posts()
    chart_html = generate_offensive_yards_chart()
    
    # Create a list of article components for each post
    article_posts = [
        Article(
            H1(post['title'], cls='post-title'),
            Div(NotStr(post['content']))
        )
        for post in posts
    ]
    return Html(
        Head(
            Title('Deep Charts: NFL Yards Tracker'),
            Link(rel='stylesheet', href='https://cdn.jsdelivr.net/npm/@picocss/pico@latest/css/pico.min.css'),
            Style("""
                .header { 
                    text-align: center; 
                    padding: 1em; 
                    background-color: #f8f9fa; 
                    position: fixed; 
                    top: 0; 
                    width: 100%; 
                    z-index: 10; 
                }
                .container { 
                    display: flex; 
                    max-width: 100%; 
                    margin-top: 80px; /* Space for the fixed header */
                }
                .posts { 
                    flex: 2; 
                    overflow-y: auto; 
                    height: calc(100vh - 80px); /* Adjust for header */
                    padding: 1em; 
                    margin-right: 40%; 
                    box-sizing: border-box; 
                }
                .chart { 
                    flex: 1; 
                    position: fixed; 
                    right: 0; 
                    top: 80px; /* Space for the fixed header */
                    width: 40%; 
                    height: calc(100vh - 80px); /* Adjust for header */
                    padding: 1em; 
                    box-sizing: border-box; 
                }
                h1.post-title { 
                    font-size: 1.5em; 
                    font-weight: bold; 
                }
                article { 
                    margin-bottom: 2em; 
                }
            """)
        ),
        Body(
            Div(
                H1('Deep Charts: NFL Yards Tracker', cls='header'),
                Div(
                    Div(*article_posts, cls="posts"),
                    Div(NotStr(chart_html), cls="chart"),
                    cls="container"
                )
            )
        )
    )



##################################
## 6. Serve the App
##################################

serve()








