import langchain
import pandas as pd
import numpy as np
import streamlit as st
from chart_generator import ChartGenerator

class LangchainProcessor:
    def __init__(self, data):
        self.data = data

    def describe_data(self):
        # Generate a brief description of the data
        description = f"This dataset contains {len(self.data)} rows and {len(self.data.columns)} columns.\n\n"

        # Describe numerical columns
        numerical_columns = self.data.select_dtypes(include=['number']).columns
        print(numerical_columns)
        if len(numerical_columns) > 0:
            numerical_description = self.data[numerical_columns].describe()#.to_string()
            description += f"The numerical columns have the following statistics:\n"
            st.write(description)
            st.dataframe(numerical_description)

        # Describe categorical columns
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        print("categorical_columns",categorical_columns)
        if len(categorical_columns) > 0:
            chart_generator = ChartGenerator(self.data)
            description = "\n\nThe categorical columns include:\n"
            for col in categorical_columns:
                unique_values = self.data[col].nunique()
                description += f"- Column '{col}' has {unique_values} unique values.\n"
                print(self.data[col].value_counts())
                chart_generator.generate_chart_col(self.data[col].value_counts(),chart_type='pie')

        # Display the generated description
        st.write(description)

    def determine_chart_type(self):
        # Use Langchain to understand the context of the data and determine the most appropriate type of chart
        # This is a placeholder. The actual implementation will depend on the specific requirements of the data and Langchain.
        chart_type = 'bar'
        return chart_type
