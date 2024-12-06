import pandas as pd
import streamlit as st
from data_processor import DataProcessor
from chart_generator import ChartGenerator
from langchain_processor import LangchainProcessor

import matplotlib.pyplot as plt

# Set Matplotlib backend to TkAgg
plt.switch_backend('TkAgg')

# # Your Streamlit code
# st.title("Matplotlib in Streamlit")
# # Create and display a Matplotlib figure
# plt.plot([1, 2, 3, 4])
# st.pyplot(plt)

def main():
    st.title('AI Chart Generator')

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data_processor = DataProcessor(data)
        cleaned_data = data_processor.preprocess()

        langchain_processor = LangchainProcessor(cleaned_data)
        langchain_processor.describe_data()
        chart_type = langchain_processor.determine_chart_type()

        chart_generator = ChartGenerator(cleaned_data)
        chart_generator.generate_chart(chart_type)

if __name__ == "__main__":
    main()
