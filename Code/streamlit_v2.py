import numpy as np 
import pandas as pd
import streamlit as st 
import openai
import os
from function import visualize_timeseries ,get_completion,yoy_growth,calculate_trend_slope_dataframe,extract_text_from_pdf,read_text_file
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pyperclip
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
            page_title="Sigmoid GenAI",
            page_icon="/Users/rahulkushwaha/Desktop/git/demand_forecasting_genai/Data/cropped-Sigmoid_logo_3x.png"  
        )
st.sidebar.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
st.sidebar.image("/Users/rahulkushwaha/Desktop/git/demand_forecasting_genai/Data/cropped-Sigmoid_logo_3x.png", use_column_width=True)
st.sidebar.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
API = st.sidebar.text_input("Enter the API key:",type="password")
if st.sidebar.button("Enter"):
    openai.api_key = API
    st.sidebar.success("API key successfully set!")

def select_country(d):
    country = st.sidebar.selectbox("Select Country:", d["geo"].unique().tolist())
    return country

def select_level(d):
    levels = ["geo", "channel", "brand", "SKU"]
    selected_levels = st.sidebar.multiselect("Select Levels", levels, default=["geo"])

    selected_channel = None
    selected_brand = None
    selected_SKU = None

    if "channel" in selected_levels:
        st.sidebar.header("Channel")
        channel_options = d["channel"].unique().tolist()
        selected_channel = st.sidebar.selectbox("Select Channel:", channel_options)

    if "brand" in selected_levels:
        st.sidebar.header("Brand")
        brand_options = d["brand"].unique().tolist()
        selected_brand = st.sidebar.selectbox("Select brand:", brand_options)

    if "SKU" in selected_levels:
        st.sidebar.header("SKU")
        SKU_options = d["SKU"].unique().tolist()
        selected_SKU = st.sidebar.selectbox("Select SKU:", SKU_options)

    return selected_levels, selected_channel, selected_brand, selected_SKU



##Reading the data
df_dash = pd.read_csv("/Users/rahulkushwaha/Desktop/git/demand_forecasting_genai/Data/Retail_Data.csv")
tab1, tab2 ,tab3= st.tabs(["About the App", "Demand forecasting interpreater","CodeAI"])
with tab2:

    def main():
        st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: blue;'>GenAI: Time Series Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.subheader("👨‍💻  How to Use")
        st.write("1. Select a country from the sidebar to filter data.")
        st.write("2. Choose the levels you want to analyze: geo, channel, brand, SKU.")
        st.write("3. Visualize your time series data.")
        st.write("4. Click on Get insights.")
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.subheader(" ⚠️  Limitations")
        st.write("- It may not capture all nuances and context")
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.sidebar.header("User Inputs")
        country = select_country(df_dash)
        selected_levels = select_level(df_dash)

        # Time Series Visualization Section
        st.subheader("Visualize your time series")
        st.markdown("---")
        data = visualize_timeseries(df_dash,selected_levels[0], country,
                                    selected_levels[1], selected_levels[2], selected_levels[3])    
        data_trend = calculate_trend_slope_dataframe(data)
        if data_trend.empty:
            pass
        else:
            data_trend_2=data_trend.groupby(["scenario","trend"])[["slope_his","slope_for"]].mean().reset_index()
        if data.empty:
            pass
        else:
            data_yoy = yoy_growth(data)
        data_trend_3 =data_trend_2[["scenario","trend"]]
        if st.button("Get Analysis"):
            ## Forescated and Historical analysis
            analysis_string ="""Generate the analysis based on instruction\
                                    that is delimated by triple backticks.\
                                    isntruction: ```{instruction_analyis}```\
                                    """
            analysis_templete= ChatPromptTemplate.from_template(analysis_string)

            instruction_analyis =f"""You are functioning as an AI data analyst.
            1.You will be analyzing two datasets: trend_dataset and year on year growth dataset.
            2.Trend_dataset has the following columns:
                Scenario: Indicates if a data point is historical or forecasted.
                Trend: Indicates the trend of the data for a specific scenario.
                year on year growth dataset has the following columns:
                Year: Indicates the year.
                yoy_growth: Indicates the percentage volume change compared to the previous year.
            4.Start the output as "Insight and Findings:" and report in point  form
            5.Analyze the trend based on the 'Trend' column of the trend_dataset:
                a.Analyze Historical Data.
                b.Analyze Forecasted Data.
            6.Analyze the year on year growth based on the year on year growth dataset also include the change percentage
            7.Provide this analysis without including any code.
            8.The datasets: {data_trend_3} for trend analysis and {data_yoy} for year-on-year growth analysis.
            9.Report back only the insights and findings.
            10.Use at most 200 words.
            11.provide conclusions about the dataset's performance over the years and include suggestions for why fluctuations occurred also include the year on year 
            12.Present your findings as if you are analyzing a plot."""
            chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0301",openai_api_key=API)
            user_analysis = analysis_templete.format_messages(instruction_analyis=instruction_analyis)
            response = chat(user_analysis)
            st.write(response.content)
        st.markdown("---")

    if __name__ == "__main__":
        main()
with tab1:
    st.header("About The App")
    st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)   
    # Add your app description and information here
    st.markdown("👋 Welcome to Sigmoid GenAI - Your Data Analysis APP!")
    st.write("This app is designed to help you analyze and visualize your data.")
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
    st.subheader("👨‍💻  How to Use")
    st.write("1. Please enter your API key in side bar and click on the ENTER")
    st.write("2. From the top this page please select the required tab")
    st.write("3. Follow the instruction of that tab.")
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
    st.subheader("⚠️   Limitations")
    st.write("Please note the following limitations:")
    st.write("- Active internet connection is required.")
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
 
 #Tab 3
with tab3:

    # Initialize an empty dictionary to store column descriptions
    column_descriptions = {}

    def main():
        st.markdown('<p style="color:red; font-size:30px; font-weight:bold;">CodeAI:</p>', unsafe_allow_html=True)
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">How to Use:</p>', unsafe_allow_html=True)
        st.markdown("""
        - 📂 Upload a CSV or Excel file containing your dataset.
        - 📝 Provide descriptions for each column of the dataset in the 'Column Descriptions' section.
        - ❓ Ask questions about the dataset in the 'Ask a question about the dataset' section.
        - 🔍 Click the 'Get Answer' button to generate answer based on your question.
        """)

        # Display limitations with emojis
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Limitations ⚠️:</p>', unsafe_allow_html=True)
        st.markdown("""
        - The quality of AI responses depends on the quality and relevance of your questions.
        - Ensure that you have a good understanding of the dataset columns to ask relevant questions.
        """)   
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Head of the Dataset:</p>', unsafe_allow_html=True)
        
        df_user = pd.DataFrame()
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_user = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                    df_user = pd.read_excel(uploaded_file)
                
                # Display the first few rows of the dataset
                st.write(df_user.head())
                

                st.info("Please add column descriptions of your dataset")
                for col in df_user.columns:
                    col_description = st.text_input(f"Description for column '{col}':")
                    if col_description:
                        column_descriptions[col] = col_description
                
                if st.button("Submit Descriptions"):
                    st.success("Descriptions submitted successfully!")
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")
                return
        
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:red; font-size:25px; font-weight:bold;">Ask a question about the dataset:</p>', unsafe_allow_html=True)
        user_question = st.text_input(" ")
        

        code_string ="""Generate the python code based on the user question\
            that is delimated by triple backticks\
                based on the instruction that is {instruction}.\
                    user question: ```{user_question}```\
                        """
        code_templete= ChatPromptTemplate.from_template(code_string)

        instruction =f"""1. You are functioning as an AI data analyst.
        2. Task: Respond to questions based on the provided dataset.
        4. Dataset columns enclosed in square brackets {df_user.columns.tolist()}.
        5. Columns Description in dict format - {column_descriptions}.
        6. Provide code based on the user's question.
        7. DataFrame name: 'df_user'.
        8. Print result using 'st.write' for text or 'st.plot' for plots.
        9. Return the output in function form only.
        10. Call the function below the response in the same script.
        11. Provide all the code together.
        12. Only return the code; no explanations or extra text.
        13. Include code to suppress warnings.
        14. Do not include [assistant].
        15. Do not read any dataset; call the function with df_user.
        16. Return final output with st.write or st.pyplot.
        17. Only give the executable code.
        18. Code must start with 'def' and end with the function call.
        19. Do not enclose the code in triple backticks.
        20. Only give the executable line; no non-executable characters."""

        user_message = code_templete.format_messages(instruction=instruction,user_question=user_question)
                
        st.markdown('<style>div.row-widget.stButton > button:first-child {background-color: blue; color: white;}</style>', unsafe_allow_html=True)
        chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0301",openai_api_key=API)
        if st.button("Get Answer"):
            if user_question:
                user_message = code_templete.format_messages(instruction=instruction,user_question=user_question)
                code = chat(user_message)
                st.code(code.content)
                exec(code.content)
            else:
                st.warning("Not a valid question. Please enter a question to analyze.")
        
        # st.markdown('<p style="color:red; font-size:25px; font-weight:bold;">Code Execution Dashboard:</p>', unsafe_allow_html=True)
    
        # st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        # code_input = st.text_area("Enter your code here", height=200)
        # st.warning(("⚠️ If there is any non-executable line in generated code; please remove it"))
        
        # if st.button("Execute code"): 
        #     try:
        #         # Use exec() to execute the code
        #         exec(code_input)
        #     except Exception as e:
        #         st.error(f"An error occurred: {e}")

    # Check if the script is run as the main program
    if __name__ == "__main__":
        main()
