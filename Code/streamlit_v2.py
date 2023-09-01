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
openai.api_key = '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'
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



    ### Chat boat
        df_user = pd.read_csv("Data/Retail_Data.csv")
        st.markdown("---")
        st.subheader("Have more Questions? \U0001F4AC")
        st.write("Head of the dataset")
        st.write(df_user.iloc[:, 1:].head())
        
        user_question = st.text_input("Ask a question about the dataset:") 
        user_prompt = f"""
        You are functioning as an AI data analyst. Your task is to respond to questions based on a provided dataset enclosed within square bracket [] .
        If a question is not related to the dataset, reply with Not a valid question in st.warning format 
        Do not start and end the code scripy with backticks for example like  this ```python and ```
        donot read the dataframe using pd.read_csv just pass the ''df_user'' as function input
        do not enclosed the code between the backticks like this ```python ```
        do not start code script like ```python and do not end like ``` code must start with function defination and end end with function call   
        Follow the instructions below and only return the executable code. The last line of the code should not be '```'; it should be the function call.
        Do not call the function like unique_countries(df_user)``` or unique_countries(df_user) at the end of the code script. Do not write the backticks at the end.
        Always show the function output with st.write or st.plot
        [user_question] - {user_question} and start the code with function defination and end with function call and show the output either with st.write or st.pyplot and The code should not be enclosed within triple backticks.
        [dataset columns] - {df_user.columns.tolist()}
        ### Dataset Columns Description ###
        'month' - Indicate the date on a monthly level. Treat it as a datetime; if needed, convert it to datetime.
        'volume' - Total sales value on a particular date.
        'year' - Indicate the year.
        'channel' - Indicate whether the product is sold online or offline.
        'brand' - Indicate the brand of the product.
        'sku' - Indicate the SKU.
        'scenario' - Indicate whether the corresponding month is a historical date or a forecasted month.
        'Region name' - Indicate the region name.
        'Market name' - Indicate the market name.
        'category' - Indicate the category.
        Provide code based on the user's instruction, keeping in mind that the name of the DataFrame is '''df_user'''.
        Also, you have to print the final result of the code using 'st.write' for text or 'st.plot' for plots.
        Return the output in function form only. Call the function below the response in the same script and provide all the code together.
        Only give the output of the function as a response. Only return the code; do not include any explanations or extra text. 
        If you include any comments, make sure each line starts with '#'.
        Also, include the code to suppress any warnings. Do not include [assistant] - 
        donot read the dataframe using pd.read_csv just pass the ''df_user'' as function input
        do not do this step df_user = pd.read_csv('dataset.csv')
        do not read any any dataset just call the function with the df_user
        Always return the final output with st.write or st.pyplot
        """



        if st.button("Get Answer"):
            if user_question:
                #ai_response = generate_response_with_api(user_question, df_user)
                ai_response = get_completion(user_prompt)
                code =st.code(ai_response)
                #exec(ai_response)            
            else:
                st.write("Please enter a question to analyze.")

        st.subheader("Code Execution Dashboard")  
        
        # Create a code input textarea
        code = st.text_area("Enter your code here", height=200)

        # Add a button to run the code
        if st.button("Execute code"): 
            try:
                # Use exec() to execute the code
                exec(code)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        st.markdown("---")
        ### PDF or Txt Summarizer
        st.subheader("Summarize Reports and Files")
        st.markdown("Upload your report or file and get a summary.")
        st.info("Please upload the file through the sidebar.")
        st.sidebar.title("File Upload")
        st.sidebar.markdown("📄 Upload a TXT or PDF file")

        uploaded_file = st.sidebar.file_uploader("",
                                                    type=["txt", "pdf"],
                                                    help="")

        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            
            file_extension = uploaded_file.name.split(".")[-1]

            if file_extension == "txt":
                content = read_text_file(uploaded_file.name)
            elif file_extension == "pdf":
                content = extract_text_from_pdf(uploaded_file.name)
            else:
                st.sidebar.error("Unsupported file format. Please upload a TXT or PDF file.")
                return
            user_question_text = st.text_area("Enter your questions here:", value="", height=100)
            prompt_file = f"**Instructions:** Answer the user's question based on the provided paragraph,report,text,file only and if the question is not from the \
                                file or text  always return complete response\
                        \n\n**Paragraph:**\n[{content}]\
                        \n\n**User Question:**\n[{user_question_text}]"

            if st.button("Get Answer from File"):
                response3 = get_completion(prompt_file)
                st.write(response3)
                pyperclip.copy(response3)
                st.success('Text copied successfully!')

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
    # Add any additional information or images as needed


    # Your existing code for the "App" tab goes here

    #write an email based on the summary of report to CEO not more than in 200 words write in point form at maximum include 10 points only in the email
