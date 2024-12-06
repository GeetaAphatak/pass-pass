import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
plt.switch_backend('TkAgg')

class ChartGenerator:
    def __init__(self, data):
        self.data = data

    def generate_chart(self, chart_type):
        # Generate the appropriate chart based on the chart_type
        # This is a placeholder. The actual implementation will depend on the specific requirements of the data and the chart_type.
        if chart_type == 'bar':
            sns.barplot(data=self.data)
        elif chart_type == 'line':
            sns.lineplot(data=self.data)
        elif chart_type == 'pie':
            self.data.plot(kind='pie')
        # plt.show()
        st.pyplot(plt)

    def generate_chart_col(self, values, chart_type):
        # Generate the appropriate chart based on the chart_type
        # This is a placeholder. The actual implementation will depend on the specific requirements of the data and the chart_type.
        if chart_type == 'bar':
            sns.barplot(data=values)
        elif chart_type == 'line':
            sns.lineplot(data=values)
        elif chart_type == 'pie':
            # values.plot(kind='pie')
            plt.figure(figsize=(8, 6))
            plt.pie(values, labels=values.index, autopct='%1.1f%%', startangle=140)
        # plt.show()
        st.pyplot(plt)
