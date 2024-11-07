import streamlit as st
import matplotlib.pyplot as plt

def customer_sentiment_page():
    st.logo('assets/logo_black.png', icon_image='assets/logo_magenta.png')    
    st.header("Customer Sentiment Analysis Results")

    if 'sentiment_df' in st.session_state:
        sentiment_df = st.session_state['sentiment_df']
        all_customer_sentiments = st.session_state['all_customer_sentiments']
        
        overall_avg_customer_sentiment = sentiment_df['sentiment_score'].mean() if not sentiment_df.empty else 0

        st.subheader(f"Average sentiment score for all customer responses: {overall_avg_customer_sentiment:.2f}/100")
        progress_value = int(overall_avg_customer_sentiment) 
        st.progress(progress_value) 

        st.dataframe(sentiment_df)


        fig2, ax2 = plt.subplots()
        ax2.hist(all_customer_sentiments, bins=10, range=(0, 100), color='skyblue', edgecolor='black')
        ax2.set_xlabel('Customer Sentiment Score (1 = Negative, 100 = Positive)')
        ax2.set_ylabel('Number of Responses')
        ax2.set_title('Customer Sentiment Distribution')
        st.pyplot(fig2)

    else:
        st.write("Please perform sentiment analysis first on the main page.")
        st.stop()

        
