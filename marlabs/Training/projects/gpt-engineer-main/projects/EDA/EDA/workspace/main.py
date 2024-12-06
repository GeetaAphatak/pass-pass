from streamlit_app import StreamlitApp
# import streamlit as st
# import pandas as pd
# st.title('AI Chart Generator')
#
# uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
if __name__ == "__main__":
    app = StreamlitApp("C:\\Users\\Geeta.Phatak\\Downloads\\test.csv")
    app.run()
