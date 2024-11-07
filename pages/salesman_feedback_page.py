import streamlit as st
from utils import get_feedback_for_salesman
from utils import read_answer


def salesman_feedback_page():
    st.logo('assets/logo_black.png', icon_image='assets/logo_magenta.png')    
    st.header("Salesman Feedback")

    if 'salesman_sentiment_df' in st.session_state and 'salesman_worst_responses' in st.session_state:
        salesman_sentiment_df = st.session_state['salesman_sentiment_df']
        salesman_worst_responses = st.session_state['salesman_worst_responses']

        salesman_list = salesman_sentiment_df['salesman_name'].unique()
        selected_salesman = st.selectbox("Select a salesman to get feedback", salesman_list)

        if selected_salesman:
            salesman_data = salesman_worst_responses[salesman_worst_responses['salesman_name'] == selected_salesman]
            responses_with_scores = list(zip(salesman_data['salesman_response'], salesman_data['sentiment_score']))

            feedback = get_feedback_for_salesman(selected_salesman, responses_with_scores)

            st.subheader(f"Feedback for {selected_salesman}")
            st.write(feedback)

            st.subheader(f"Worst responses for {selected_salesman}:")
            for idx, (resp, score) in enumerate(responses_with_scores, 1):
                with st.expander(f"Response {idx}"):
                    st.write(f"{idx}. Sentiment Score: {score}")
                    st.write(f"Response: {resp}")

            read_answer(feedback)


    else:
        st.write("Please perform sentiment analysis first on the main page.")
        st.stop()