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
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
            page_title="Sigmoid GenAI",
            page_icon="Code/cropped-Sigmoid_logo_3x.png"  
        )

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
df = pd.read_csv("Data/Retail_Data.csv")
tab1, tab2 = st.tabs(["About the App", "App"])
with tab2:
    st.header("The APP")
    def main():
            # Set a custom page title and icon
        st.sidebar.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.sidebar.image("Code/cropped-Sigmoid_logo_3x.png", use_column_width=True)
        st.sidebar.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
        API = st.sidebar.text_input("Enter the API key:")
        if st.sidebar.button("Enter"):
            openai.api_key = API
            st.sidebar.success("API key successfully set!")
        st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: blue;'>GenAI Data Analysis Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.sidebar.header("User Inputs")
        country = select_country(df)
        selected_levels = select_level(df)

        # Time Series Visualization Section
        st.subheader("Visualize your time series")
        st.markdown("---")
        data = visualize_timeseries(df,selected_levels[0], country,
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
        ## Forescated and Historical analysis
        if st.button("Get Analysis"):
            prompt = f"You are functioning as an AI data analyst.You will be answer based on two dataset trend_dataset and year on year growth dataset \
            Trend datset have following columns:\n\n\
            - **Scenario**: An indicator that distinguishes whether a given volume data point is of historical nature or a forecast. \
            Specifically, the 'historical' scenario signifies historical data, while the 'forecasted' scenario points to forecasted data.\n\n\
            - **Trend**: Indicate the trend of the data for that speific scenario\
            Year on year growth dataset have following columns:\
            - **Year**: Indicate yera\
            - ** yoy_growt** : Indicate percenatge volume chnaged with respect to previous year\
            Befor performing the task check wheter the dataset have data or not If dataset is empty, \
            donot perform the other task return a message in bold letters that '''There is no dataset to perfrom analysis'''\
            Start the ouput as '''Inshight and Findings:-''' and also give the output in the point format\
            The primary objectives encompass the following tasks:\n\n\
            Analyze the trend only on the base of ''trend column'' of  trend dataset\
            1. **Analyze Historical Data**\n\n\
            2. **Analyze Forecasted Data** \n\n\
            On the basis of year on year growth dataset do the following:-\
            - Conclusions about the dataset's performance over the years. And also include some suggestions why the flacuation have occurs \n\n\
            Please provide this analysis without the inclusion of any code. \
            The provided trend_dataset,  {data_trend_3}\n\n\
            The provide year on year growth dataset, {data_yoy}\
            Please only report back the Inshigt and findings\
            Use at most 200 words. For each analyis return only one line and donot return any note\
            Do not include the name like trend dataset or year on year growth dataset ,write your response such that your are giving your response\
            by analysing a plot\
            "
            response = get_completion(prompt)
            st.write(response)
        st.markdown("---")
        # ### PDF or Txt Summarizer
        # st.subheader("Summarize Reports and Files")
        # st.markdown("Upload your report or file and get a summary.")
        # st.info("Please upload the file through the sidebar.")
        # st.sidebar.title("File Upload")
        # st.sidebar.markdown("📄 Upload a TXT or PDF file")

        # uploaded_file = st.sidebar.file_uploader("",
        #                                             type=["txt", "pdf"],
        #                                             help="")

        # if uploaded_file is not None:
        #     st.success("File uploaded successfully!")
            
        #     file_extension = uploaded_file.name.split(".")[-1]

        #     if file_extension == "txt":
        #         content = read_text_file(uploaded_file.name)
        #     elif file_extension == "pdf":
        #         content = extract_text_from_pdf(uploaded_file.name)
        #     else:
        #         st.sidebar.error("Unsupported file format. Please upload a TXT or PDF file.")
        #         return
        #     user_question_text = st.text_area("Enter your questions here:", value="", height=100)
        #     prompt_file = f"**Instructions:** Answer the user's question based on the provided paragraph,report,text,file only and if the question is not from the \
        #                         file or text  always return complete response\
        #                 \n\n**Paragraph:**\n[{content}]\
        #                 \n\n**User Question:**\n[{user_question_text}]"

        #     if st.button("Get Answer from File"):
        #         response3 = get_completion(prompt_file)
        #         st.write(response3)
        #         pyperclip.copy(response3)
        #         st.success('Text copied successfully!')

    if __name__ == "__main__":
        main()
with tab1:
    st.header("About The App")
    st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)   
    # Add your app description and information here
    st.markdown("👋 Welcome to Sigmoid GenAI - Your Data Analysis Dashboard!")
    st.write("This app is designed to help you analyze and visualize your data.")
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
    st.subheader("How to Use")
    st.write("1. Select a country from the sidebar to filter data.")
    st.write("2. Choose the levels you want to analyze: geo, channel, brand, SKU.")
    st.write("3. Visualize your time series data.")
    st.write("4. Analyze historical and forecasted data trends.")
    st.write("5. Get insights on year-on-year growth.")
    st.write("6. Ask questions about the dataset using the chatbot.")
    st.write("7. Summarize reports and files.")
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
    st.subheader("🎯  Limitations")
    st.write("Please note the following limitations:")
    st.write("- Limited to time series data analysis and visualization.")
    st.write("- Complex data handling may require additional coding.")
    st.write("- Limited file format support for summarization (TXT and PDF).")
    st.write("- Active internet connection is required.")
    st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
 
 #Tab 3
with tab3:

    # Initialize an empty dictionary to store column descriptions
    column_descriptions = {}

    # Create a function to define the main content of your Streamlit app
    def main():
        st.markdown('<p style="color:red; font-size:30px; font-weight:bold;">CodeAI:</p>', unsafe_allow_html=True)
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">How to Use:</p>', unsafe_allow_html=True)
        st.markdown("""
        - 📂 Upload a CSV or Excel file containing your dataset.
        - 📝 Provide descriptions for each column of the dataset in the 'Column Descriptions' section.
        - ❓ Ask questions about the dataset in the 'Ask a question about the dataset' section.
        - 🔍 Click the 'Get Answer' button to generate code based on your question.
        - ▶️ View and execute the generated code in the 'Code Execution Dashboard' section; please remove if any non-executable line is generated
        """)

        # Display limitations with emojis
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Limitations ⚠️:</p>', unsafe_allow_html=True)
        st.markdown("""
        - The quality of AI responses depends on the quality and relevance of your questions.
        - Ensure that you have a good understanding of the dataset columns to ask relevant questions.
        - 🔒 Security: Be cautious when executing code, as it allows arbitrary code execution.
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
                
                # Prompt the user to add column descriptions
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
        
        # Generate a user prompt with dataset information
        user_prompt = f"""
        You are functioning as an AI data analyst. Your task is to respond to questions based on the provided dataset.
        The user question will be delimited by single quote '{user_question}' , and the columns of the dataset are enclosed in square brackets {df_user.columns.tolist()}.
        Dataset Columns Description is enclosed in dict format - {column_descriptions}.
        Provide code based on the user's question, keeping in mind that the name of the DataFrame is 'df_user'.
        Also, you have to print the final result of the code using 'st.write' for text or 'st.plot' for plots.
        Return the output in function form only. Call the function below the response in the same script and provide all the code together.
        Only give the output of the function as a response. Only return the code; do not include any explanations or extra text.
        If you include any comments, make sure each line starts with '#'.
        Also, include the code to suppress any warnings. Do not include [assistant].
        Do not read any dataset; just call the function with the df_user.
        Always return the final output with st.write or st.pyplot.
        Only give the code according to the user question
        Do not enclose the code in triple backticks only give the executable code the code must start with the function def and end with the function call\n
        Only give the executable line do not include any character that is non-executable
        """
        
        st.markdown('<style>div.row-widget.stButton > button:first-child {background-color: blue; color: white;}</style>', unsafe_allow_html=True)
        
        if st.button("Get Answer"):
            if user_question:
                ai_response = get_completion(user_prompt)
                code = st.code(ai_response)
            else:
                st.warning("Not a valid question. Please enter a question to analyze.")
        
        st.markdown('<p style="color:red; font-size:25px; font-weight:bold;">Code Execution Dashboard:</p>', unsafe_allow_html=True)
    
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        code_input = st.text_area("Enter your code here", height=200)
        st.warning(("⚠️ If there is any non-executable line in generated code; please remove it"))
        
        if st.button("Execute code"): 
            try:
                # Use exec() to execute the code
                exec(code_input)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Check if the script is run as the main program
    if __name__ == "__main__":
        main()
