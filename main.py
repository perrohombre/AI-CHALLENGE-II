import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

# Import shared functions from utils.py
from utils import (
    analyze_sentiment_customer,
    analyze_sentiment_salesman,
    extract_customer_responses,
    extract_salesman_responses,
    load_data,
    get_feedback_for_salesman
)

logo_path = "assets/logo.png"

def format_time(seconds):
    # Convert seconds into hours, minutes, and seconds
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)

    if hours > 0:
        return f"{int(hours)}h {int(mins)}m {int(secs)}s"
    elif mins > 0:
        return f"{int(mins)}m {int(secs)}s"
    else:
        return f"{int(secs)}s"

# main.py
def main_page():
    st.logo('assets/logo_black.png', icon_image='assets/logo_magenta.png')

    st.title("AI CHALLENGE II")

    # Load data to determine the maximum number of conversations
    data = load_data()
    max_rows = len(data)

    # Add a slider to select the number of conversations to analyze
    num_rows = st.slider("Select number of conversations to analyze", min_value=1, max_value=max_rows, value=min(100, max_rows))

    average_time_per_conversation = 4
    # Estimate total time based on the selected number of conversations
    estimated_total_time = average_time_per_conversation * num_rows
    # Format the estimated time into a human-readable format
    formatted_time = format_time(estimated_total_time)
    # Display the estimated time
    st.write(f"⏱ {formatted_time}")

    if 'sentiment_df' not in st.session_state:
        if st.button("Analyze Sentiments"):
            data = data.head(num_rows)  # Use the selected number from the slider

            # Initialize variables to store results
            sentiment_results = []
            all_customer_sentiments = []
            all_salesman_sentiments = []
            salesman_sentiment_results = []

            # Initialize the progress bar
            progress_bar = st.progress(0)
            total_conversations = len(data)  # Total number of conversations for progress tracking

            # Create two columns for side-by-side plots
            col1, col2 = st.columns(2)

            # Create placeholders for charts
            customer_sentiment_chart = col1.empty()
            salesman_sentiment_chart = col2.empty()

            # Initialize the figures outside the loop so they are reused
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()

            for i, row in data.iterrows():
                # Extract responses
                customer_responses = extract_customer_responses(row['conversation'])
                salesman_responses = extract_salesman_responses(row['conversation'])

                # Analyze customer responses
                for response in customer_responses:
                    print("customer")
                    sentiment_score = analyze_sentiment_customer(response)
                    sentiment_results.append({
                        'customer_response': response,
                        'sentiment_score': sentiment_score
                    })
                    all_customer_sentiments.append(sentiment_score)

                # Analyze salesman responses
                for response in salesman_responses:
                    print("salesman")
                    sentiment_score = analyze_sentiment_salesman(response)
                    salesman_sentiment_results.append({
                        'salesman_name': row['salesman_name'],
                        'salesman_response': response,
                        'sentiment_score': sentiment_score
                    })
                    all_salesman_sentiments.append(sentiment_score)

                

                # Update progress bar
                progress_bar.progress((i + 1) / total_conversations)

                # Clear and update customer sentiment histogram
                ax1.clear()
                ax1.hist(all_customer_sentiments, bins=10, range=(0, 100), color='skyblue', edgecolor='black')
                ax1.set_xlabel('Customer Sentiment Score (1 = Negative, 100 = Positive)')
                ax1.set_ylabel('Number of Responses')
                ax1.set_title('Customer Sentiment Distribution')
                customer_sentiment_chart.pyplot(fig1)

                # Clear and update salesman sentiment histogram
                ax2.clear()
                ax2.hist(all_salesman_sentiments, bins=10, range=(0, 100), color='lightcoral', edgecolor='black')
                ax2.set_xlabel('Salesman Sentiment Score (1 = Negative, 100 = Positive)')
                ax2.set_ylabel('Number of Responses')
                ax2.set_title('Salesman Sentiment Distribution')
                salesman_sentiment_chart.pyplot(fig2)

            sentiment_df = pd.DataFrame(sentiment_results)
            salesman_sentiment_df = pd.DataFrame(salesman_sentiment_results)

            # Find 10 worst responses for each salesman
            salesman_worst_responses = salesman_sentiment_df.groupby(
                ['salesman_name']
            ).apply(lambda x: x.nsmallest(10, 'sentiment_score')).reset_index(drop=True)

            # Store results in session state
            st.session_state['sentiment_df'] = sentiment_df
            st.session_state['salesman_sentiment_df'] = salesman_sentiment_df
            st.session_state['all_customer_sentiments'] = all_customer_sentiments
            st.session_state['all_salesman_sentiments'] = all_salesman_sentiments
            st.session_state['salesman_worst_responses'] = salesman_worst_responses

            st.success("Sentiment analysis complete!")
            # Ensure the progress bar reaches 100% after completion
            progress_bar.progress(100)
        else:
            st.write("Please click the 'Analyze Sentiments' button to perform the analysis.")
            st.stop()
    else:
        st.write("Sentiment analysis already performed. Navigate to other pages to see results.")

# Set up the pages and navigation
def main():
    # Import page functions from the pages directory
    from pages.customer_sentiment_page import customer_sentiment_page
    from pages.salesman_sentiment_page import salesman_sentiment_page
    from pages.salesman_feedback_page import salesman_feedback_page
    from pages.call_real_time_analysis import call_real_time_analysis
    from pages.RAG_page import RAG_page

    # Create page objects
    main_pg = st.Page(main_page, title="Home", icon="🏠", default=True)
    customer_sentiment_pg = st.Page(customer_sentiment_page, title="Customer Sentiment", icon="🙂")
    salesman_sentiment_pg = st.Page(salesman_sentiment_page, title="Salesman Sentiment", icon="🧑‍💼")
    salesman_feedback_pg = st.Page(salesman_feedback_page, title="Salesman Feedback", icon="📝")
    call_real_time_analysis_pg = st.Page(call_real_time_analysis, title="Call Support center", icon="📞")
    RAG_page = st.Page(RAG_page, title="RAG", icon="🔍")

    # Set up navigation
    pg = st.navigation(
    {
        "": [main_pg],
        "Use Case I": [customer_sentiment_pg, salesman_sentiment_pg,
                       salesman_feedback_pg, call_real_time_analysis_pg],
        "Use Case II": [
            RAG_page
        ]
    }
    )

    st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="📊")
    pg.run()

if __name__ == "__main__":
    main()