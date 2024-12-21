import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import gdown



# Set page config
st.set_page_config(page_title="Gaming Dataset Analysis", layout="wide")

# Function to load and preprocess data
@st.cache_data
import pandas as pd
import gdown

def load_data():
    # Google Drive file ID
    file_id = "1oN4uVE6VmNEiOJlA-SMWD4AFToEc2p9l"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "game_data_all.csv"

    # Download the file
    gdown.download(url, output, quiet=False)

    # Load the data into a DataFrame
    df = pd.read_csv(output)

    # Convert date columns to datetime
    df['release'] = pd.to_datetime(df['release'], errors='coerce')
    df['all_time_peak_date'] = pd.to_datetime(df['all_time_peak_date'], errors='coerce')

    # Calculate review ratio
    df['review_ratio'] = pd.to_numeric(df['positive_reviews'], errors='coerce') / pd.to_numeric(df['total_reviews'], errors='coerce')

    # Drop rows with NaN values in essential columns
    df = df.dropna(subset=['release', 'positive_reviews', 'total_reviews'])

    # Ensure specific columns are strings
    df['primary_genre'] = df['primary_genre'].astype(str)
    df['publisher'] = df['publisher'].astype(str)

    return df

try:
    df = load_data()

    st.title("🎮 Gaming Dataset Analysis Dashboard")
    st.write("Interactive analysis of gaming statistics and trends")

    # Enhanced Sidebar Filters (Dynamic Queries)
    st.sidebar.header("Filters")
    
    # Genre filter with multi-select
    genres = sorted(df['primary_genre'].unique().tolist())
    selected_genres = st.sidebar.multiselect('Select Primary Genres', genres, default=genres[0])
    
    # Publisher filter with search
    publishers = sorted(df['publisher'].unique().tolist())
    selected_publishers = st.sidebar.multiselect('Select Publishers', publishers)
    
    # Date range filter
    min_date = df['release'].min().date()
    max_date = df['release'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Measure selection for visualization
    measure_options = {
        'Peak Players': 'peak_players',
        'Total Reviews': 'total_reviews',
        'Positive Reviews': 'positive_reviews',
        'Rating': 'rating'
    }
    selected_measure = st.sidebar.selectbox('Select Measure to Visualize', list(measure_options.keys()))
    measure_column = measure_options[selected_measure]

    # Filter data based on selections
    filtered_df = df.copy()
    if selected_genres:
        filtered_df = filtered_df[filtered_df['primary_genre'].isin(selected_genres)]
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    filtered_df = filtered_df[
        (filtered_df['release'].dt.date >= date_range[0]) &
        (filtered_df['release'].dt.date <= date_range[1])
    ]

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Games", len(filtered_df), f"Selected out of {len(df)} total")
    with col2:
        measure_avg = filtered_df[measure_column].mean()
        st.metric(f"Average {selected_measure}", f"{measure_avg:,.2f}")
    with col3:
        avg_rating = filtered_df['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}%")

    # Interactive Time Series Plot with Brush Selection
    st.subheader(f"{selected_measure} Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['release'],
        y=filtered_df[measure_column],
        mode='markers',
        marker=dict(
            size=filtered_df['total_reviews'] / filtered_df['total_reviews'].max() * 20,
            color=filtered_df['rating'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Rating')
        ),
        text=filtered_df['game'],
        hovertemplate="""
        Game: %{text}<br>
        Release Date: %{x}<br>
        """ + f"{selected_measure}" + """: %{y:,.0f}<br>
        Rating: %{marker.color:.1f}%<br>
        <extra></extra>
        """
    ))
    fig.update_layout(
        dragmode='zoom',
        xaxis_title='Release Date',
        yaxis_title=selected_measure,
        hovermode='closest'
    )
    # Add range slider and buttons for time series navigation
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Reset Zoom",
                         method="relayout",
                         args=[{"xaxis.range": None, "yaxis.range": None}])
                ]
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interactive Genre Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Genre Performance")
        # Allow selecting metric for comparison
        genre_metric = st.selectbox(
            'Select metric for genre comparison',
            ['peak_players', 'total_reviews', 'rating', 'review_ratio'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        genre_data = filtered_df.groupby('primary_genre')[genre_metric].mean().sort_values()
        fig = px.bar(
            genre_data,
            orientation='h',
            title=f'Average {genre_metric.replace("_", " ").title()} by Genre',
            labels={'value': genre_metric.replace('_', ' ').title(), 'primary_genre': 'Genre'}
        )
        fig.update_traces(hovertemplate='Genre: %{y}<br>Value: %{x:,.2f}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Technology Distribution")
        # Interactive technology analysis with selection
        tech_list = filtered_df['detected_technologies'].str.split(',').explode()
        tech_counts = tech_list.value_counts()
        min_tech_count = st.slider('Minimum technology usage count', 1, int(tech_counts.max()), 5)
        tech_counts = tech_counts[tech_counts >= min_tech_count]
        
        fig = px.treemap(
            names=tech_counts.index,
            parents=['Technology' for _ in range(len(tech_counts))],
            values=tech_counts.values,
            title='Technology Usage Distribution'
        )
        fig.update_traces(hovertemplate='Technology: %{label}<br>Count: %{value}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)

    # Interactive Data Table with Sorting and Filtering
    st.subheader("Detailed Game Analysis")
    
    # Column selector for table
    default_columns = ['game', 'publisher', measure_column, 'rating', 'total_reviews']
    selected_columns = st.multiselect(
        'Select columns to display',
        df.columns.tolist(),
        default=default_columns
    )
    
    # Search functionality
    search_term = st.text_input('Search games')
    
    # Filter and sort data
    display_df = filtered_df[selected_columns]
    if search_term:
        display_df = display_df[display_df['game'].str.contains(search_term, case=False)]
    
    # Format and display table
    st.dataframe(
        display_df.style.format({
            'peak_players': '{:,.0f}',
            'total_reviews': '{:,.0f}',
            'rating': '{:.1f}%',
            'review_ratio': '{:.2%}'
        }),
        use_container_width=True,
        height=400
    )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please ensure your data file is properly formatted and contains all required columns.")
