import streamlit as st
import matplotlib.pyplot as plt

def salesman_sentiment_page():
    st.logo('assets/logo_black.png', icon_image='assets/logo_magenta.png')    
    st.header("Salesman Sentiment Analysis Results")

    if 'salesman_sentiment_df' in st.session_state:
        salesman_sentiment_df = st.session_state['salesman_sentiment_df']
        all_salesman_sentiments = st.session_state['all_salesman_sentiments']

        overall_avg_salesman_sentiment = salesman_sentiment_df['sentiment_score'].mean() if not salesman_sentiment_df.empty else 0
        st.subheader(f"Average sentiment score for all salesman responses: {overall_avg_salesman_sentiment:.2f}/100")
        progress_value = int(overall_avg_salesman_sentiment) 
        st.progress(progress_value) 

        st.dataframe(salesman_sentiment_df)

        benchmark_df = salesman_sentiment_df.groupby(
            ['salesman_name']
        )['sentiment_score'].mean().reset_index()
        benchmark_df = benchmark_df.sort_values('sentiment_score', ascending=False)

        st.subheader("Salesman Sentiment Benchmark")
        st.dataframe(benchmark_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(benchmark_df['salesman_name'], benchmark_df['sentiment_score'], color='orange')
        ax.set_xlabel('Salesman')
        ax.set_ylabel('Average Sentiment Score')
        ax.set_title('Average Sentiment Score per Salesman')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        fig3, ax3 = plt.subplots()
        ax3.hist(all_salesman_sentiments, bins=10, range=(0, 100), color='green', edgecolor='black')
        ax3.set_xlabel('Salesman Sentiment Score (1 = Negative, 100 = Positive)')
        ax3.set_ylabel('Number of Responses')
        ax3.set_title('Salesman Sentiment Distribution')
        st.pyplot(fig3)


    else:
        st.write("Please perform sentiment analysis first on the main page.")
        st.stop()