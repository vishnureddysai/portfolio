import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import base64
import pandas as pd
import datetime
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
sec_key = "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"
st.set_page_config(page_title="Vishnu Portfolio",layout="wide",initial_sidebar_state="expanded",)
repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)

st.title('Vishnu Sai Vardhan Reddy Basi R📝')
st.markdown('Design & Develop using Streamlit & Python')


logs = pd.read_csv(".//vishnu_portfolio_logs.csv")

def llm_answer(question):
    template = """
    You are a assitant which answers below question using below context
    answer based on question only.
    Question: {question}
    context is vishnu sai vardhan reddy basi r adress is 7/64-A, S N Colony, Rayachoty, Ysr District, 516269, A.P, India\r\nEmail: vishnupuma18@gmail.com; Phone: +918688353507; my hobbies are playing basketball, cricket and pubg and watches nba my fav team is GSW and players are stepehn curry and msdhoni \r\nOBJECTIVE\r\nData Scientist with 1.5+ years of professional experience and expertise in Python and machine learning, aiming to\r\npursue a Master's in Data Science with a concentration in AI to gain knowledge on advanced AI algorithms and later\r\nwork as an ML/AI Engineer.\r\nACADEMIC QUALIFICATION\r\nBachelors in Computer Science Engineering(Specialization in Data Science) CGPA: 8.36/10 Jul 2019-May 2023\r\nVellore Institute of Technology(VIT), Vellore, Tamil Nadu, India\r\nACADEMIC PROJECTS\r\nFood Recognition and Calorie Counting Using CNN [VIT, Vellore] January 2022 \xe2\x80\x93 June 2022\r\nTeam Size: 3 Role: Team Lead\r\nSummary: Worked in a three-member team to create an application that recognizes food images and calculates calorie\r\ncontent using Convolutional Neural Networks (CNN). Developed algorithms for image recognition and calorie\r\nestimation, leveraging NLP and fuzzy logic for accurate database matching. Achieved 88% accuracy in food\r\nrecognition and 98% accuracy in database matching. The project later garnered interest from a fitness club for\r\nreal-world usage. Tools used included Keras, TensorFlow, Python, and Stream-lit for the app's interactive\r\ndemonstration.\r\nDynamic Intraday Stock Price Prediction with LSTM [VIT, Vellore] August 2021 - October 2021\r\nSummary: Spearheaded an end-to-end project to predict intraday stock prices using Long Short-Term Memory\r\n(LSTM) networks. Gathered real-time data via the Yahoo API and performed comprehensive data science processes in\r\nR, from preprocessing to model training. Developed insightful visualizations using ggplot and Plotly to communicate\r\ntrends and predictions effectively, achieving an error rate of just 3%.\r\nWORK EXPERIENCE\r\nData Scientist, Maruti Suzuki India Limited, Bangalore, India June 2023 - Present\r\n\xe2\x97\x8f Currently enhancing the digital data science project\xe2\x80\x99s accuracy by fine-tuning an open-source LLM model.\r\n\xe2\x97\x8f Collaborating with vehicle domain teams like QA and EV to understand and build use cases.\r\nProject: AI-Powered Digital Data Science Team\r\n\xe2\x97\x8f Developed a system of AI agents that functioned as a digital data science team, including an AI manager, and\r\nused SQL, Python, and Spark developers to handle ETL, data analysis, and visualizations.\r\n\xe2\x97\x8f Built a Streamlit web app for user interaction, using the AI system as the backend, significantly reducing data\r\nanalysis request times from 10 days to 10 minutes.\r\n\xe2\x97\x8f The system reduced the strain on human developers and increased overall team productivity.\r\n\xe2\x97\x8f Received a quarterly performance award after demonstrating the project to the Vice President due to its\r\ntransformative impact on data processing at the company\r\n\r\n\r\nProject: GenAI-based Chatbot for Automated Data Retrieval and Visualization\r\n\xe2\x97\x8f Developed a GenAI-based chatbot that converted natural language requests into SQL queries, retrieved data\r\nfrom the database, and generated visualizations.\r\n\xe2\x97\x8f Implemented a RAG (Retrieval-Augmented Generation) system to understand the database, reducing the\r\nburden on the data team and streamlining query handling.\r\n\xe2\x97\x8f The system enabled the team to track user queries, providing better insights into data usage trends and helping\r\ndrive collaborative business decisions.\r\n\xe2\x97\x8f Achieved an accuracy rate of 97%, significantly improving the team's ability to process and analyze data.\r\nTECHNICAL SKILLS\r\n\xe2\x97\x8f Programming Languages: Python, SQL, C++, Java, R.\r\n\xe2\x97\x8f Frameworks & Libraries: Streamlit, RAG (Retrieval-Augmented Generation), Pandas.\r\n\xe2\x97\x8f AI & Machine Learning: Azure OpenAI, Generative AI, PyTorch, Scikit-Learn, NLP.\r\n\xe2\x97\x8f Data Engineering: ETL, Data Analysis, Data Visualization(matplotlib, plotly, ggplot).\r\n\xe2\x97\x8f Cloud Platforms: Microsoft Azure, IBM data pack, AWS Sagemaker & Bedrock, Databricks.\r\n\xe2\x97\x8f Tools & Technologies: Agency Swarm, SQL Server, Power BI (for visualizations).\r\n\xe2\x97\x8f Collaboration & Deployment: Git, Docker, CI/CD, Streamlit web applications\r\n\r\n\r\nINTERNSHIPS\r\nData Science Intern, Maruti Suzuki India Limited, Bangalore, India January 2023 \xe2\x80\x93 June 2023\r\nProject: Accident Detection Using LSTM Autoencoder\r\n\xe2\x97\x8f Developed a predictive model for accident detection utilizing LSTM (Long Short-Term Memory)\r\nautoencoders to analyze sequential data and identify anomalies.\r\n\xe2\x97\x8f Implemented feature engineering techniques to enhance model accuracy and efficiency, ensuring reliable\r\ndetection of potential accident scenarios.\r\n\xe2\x97\x8f Achieved an F1-score of 82, indicating the model's reliability and making it suitable for deployment.\r\n\xe2\x97\x8f Analyzed results to identify accident hotspots, contributing valuable insights to enhance vehicle safety\r\nprotocols and real-time monitoring.\r\n\xe2\x97\x8f Tools & Frameworks: Python, IBM cp4d, Keras, plotly, Scikit-learn, multi-processing.\r\nCERTIFICATIONS/WORKSHOPS/ADDITIONAL COURSES:.\r\n\xe2\x97\x8f Attended a two-day workshop at the AWS Bangalore office, focusing on AWS cloud platforms (SageMaker\r\nand Bedrock). - April 2024.\r\n\xe2\x97\x8f Developing AI Applications for Azure Coursera - December 2021 - March 2022.\r\n\xe2\x97\x8f DSA + Dynamic Programming with C++ Greek For Geeks - January 2021 - June 2021.\r\nACHIEVEMENTS/CO-CURRICULAR/EXTRACURRICULAR\r\n\xe2\x97\x8f Achieved 5 stars in both Python & SQL for consistently solving problems - HackerRank - October 2022.\r\n\xe2\x97\x8f Secured third position in the Volleyball Cluster tournament, Hyderabad - 2016-17.\r\n\xe2\x97\x8f Received gold medal for two consecutive years at the International Math Olympiad in school & participated at\r\nsecond level which was held nationwide - 2016 & 2017.\r\n\xe2\x97\x8f Participated in a national basketball tournament in the year - SVIS - 2014-15.
    Answer: give short and crisp answer."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain=LLMChain(llm=llm,prompt=prompt,)
    llm_sol = (llm_chain.invoke(question))

    return llm_sol['text']


def load_lottieurl(url):

    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def local_css(file_name):

    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style",unsafe_allow_html=True)
local_css(".//style.css")

lottie_coder = load_lottieurl("https://lottie.host/aa1ec202-5d51-4c3f-a088-d28ab42232ef/t7Wq4P0HbL.json")
lottie_python = load_lottieurl("https://lottie.host/1c1a5147-ca6d-4365-b4ce-8141a19cce33/BnvYprZxXL.json")
lottie_ds = load_lottieurl("https://lottie.host/a0d4a5f8-7d6f-417a-be21-73e7456d7410/xiV9yd2BOO.json")
lottie_skills = load_lottieurl("https://lottie.host/f65522d1-d86f-4eb7-9c77-9598dab20bcf/0I6l7LGKgT.json")
lottie_email = load_lottieurl("https://lottie.host/05b2cdde-e69e-4a6b-b134-348f22402a19/L8GZf08MHP.json")
chatbot_image = Image.open("./chatbot.PNG")
crash_detecttion_image = Image.open("./crashdetection.PNG")
profile_image = Image.open("./profilepic.JPG")
agent_image = Image.open("./agents_new.JPG",)
foodrecognization_image = Image.open("./foodrecognization.JPG",)


with st.sidebar:

    file_ = open("./profilepic.JPG","rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(f'''
        <img src="data:image/gif;base64,{data_url}" width="300" height="300" style="border-radius: 50%; border:5px solid black"/>
        ''',

        unsafe_allow_html=True
    )

    st.write("\n\n\n")
    
    st.markdown('''
    - ### [Introduction](#Intro) 
                
    - ### [Skills](#skills) 
                
    - ### [Projects](#projects) 

    - ### [Get In Touch With Me!](#msg)

    - ### [Contact](#contacts)
''')
    

    st.write("My Assistant")
    messages = st.container(height=300)
    if prompt := st.chat_input("Ask if u need more information abt me.. my work or hobbies etc"):
        messages.chat_message("user").write(prompt)
        vishnu_info = llm_answer(prompt)
      
        messages.chat_message("assistant").write(vishnu_info)
      
        messages.chat_message("assistant").write(prompt)

    st.markdown('<p>All rights reserved &#xA9; Vishnu Sai.</p>',unsafe_allow_html=True)



c1,c2,c3 = st.columns([.05,.9,.05])



with c2:

    st.divider()
    
    st.header("Introduction",anchor="Intro")
    st.write(":heavy_minus_sign:" * (12))

    c21,c22 = st.columns(2)

    with c21:
        st.write("Hello,",)
        st.markdown("### I'm Vishnu Sai!")
        st.markdown("###### Building *AI Agents* | Playing with *LLM's*",)
        st.write("An **AI Engineer** with a strong passion for artificial intelligence (AI) and machine learning (ML).My interests include Artificial Intelligence,Machine Learning,Computer Vision and Natural Language Processing.")
        st.write(' ')

        with open("./Resume vishnu masters final.pdf", "rb") as file:

            st.download_button(
                label="Resume",
                data=file,
                file_name="vishnu_resume.pdf",
            )
    with c22:
        st_lottie(lottie_coder,height=300,width=400)

   

    st.divider()



    st.header("Skills!!!",anchor="skills")
    st.write(":heavy_minus_sign:" * (7))

    c31,c32 = st.columns(2)

    with c31:

        st.markdown(""" 
                - **Technologies:**\n
                    <span style="font-size: 16px;">Machine Learning, Deep Learning, Artificial Intelligence, Web Development,Natural Language Processing, Computer Vision</span>
                  
                """,unsafe_allow_html=True
            )

        st.markdown(""" 
                - **Languages:**\n
                    <span style="font-size: 16px;">Python, C, C++, Java, SQL, R</span>
                  
                """,unsafe_allow_html=True
            )
        st.markdown(""" 
                - **Frameworks:**\n
                    <span style="font-size: 16px;">TensorFlow, Scikit, NLTK, Transformers, SpaCy, Keras, Flask, Sreamlit, Pandas, Numpy,Pytorch, Agency Swarm</span>
                  
              
                """,unsafe_allow_html=True
            )
        st.markdown(""" 
                - **Tools:**\n
                    <span style="font-size: 16px;">Azure, Aws, IBM Cloud Pack, Hugging Face, Open AI, Databricks</span>
                  
                """,unsafe_allow_html=True
            )
        st.markdown(""" 
                - **Soft Skills:**\n
                    <span style="font-size: 16px;">Leadership, Creativity, Writing, Public Speaking, Time Management, Problem Solving, Communications</span>
                  
                """,unsafe_allow_html=True
            )


    with c32:
       
        
        st_lottie(lottie_ds,height=500,width=600)
        # st_lottie(lottie_skills,height=200,width=500)



    st.divider()
    

    


    st.header("Projects",anchor="projects")
    st.write(":heavy_minus_sign:" * (7))
    col3,col4 = st.columns([1,3])

    with col3:

        st.write("")
        st.write("")
        st.image(agent_image,caption="Multi-AI Agents",use_container_width=False,)

    with col4:

            st.title("AI-Powered Digital Data Science Team",)
            st.write("A Multi AI Agent system to automate the analysis.")
            st.markdown("""
            - Description:
                - Developed a system of AI agents that functioned as a digital data science team, including an AI manager, and used SQL, Python, and Spark developers to handle ETL, data analysis, and visualizations.
                - Built a Streamlit web app for user interaction, using the AI system as the backend, significantly reducing data analysis request times from 10 days to 10 minutes.
                - Developed a RAG system of data catalogue.
                - The system reduced the strain on human developers and increased overall team productivity.
                - Received a quarterly performance award after demonstrating the project to the Vice President due to its transformative impact on data processing at the company.
            - skills used:
                - python
                - *(GEN-AI)*
            - Tools used:
                - Azure Open AI
                - VSCode
            - Frameworks used:
                - Langchain
                - Agency Swarm

    """,)
    st.divider()

    col5,col6 = st.columns([1,3])
        
    with col5:
        st.write("")
        st.write("")
        st.image(chatbot_image,caption="Chatbot",use_container_width=False,)

    with col6:
                
        st.title("Natural Language to SQL Chatbot.",)
        st.write("Chabot made with Generative AI which has trained an LLM to convert Natural Language to SQL code thorugh which runs in the database & generate output and its analysis.")
        st.markdown("""
        - Description:
            - 1.Converts Natural Question to sql query.
            - 2.developed customized agents to generate analysis/visualizatios based on the output from sql query results.
        - skills used:
            - python
            - sql
            - Deep Learning *(Transformer Model)*
        - Tools used:
            - Jupyter Notebook
            - Microsoft Azure
        - Frameworks used:
            - Streamlit
            - Azure Open AI
            - Langchain

""")

    st.divider()

    

    col7,col8 = st.columns([1,3])
        
    with col7:
        st.write("")
        st.write("")

        st.image(crash_detecttion_image,caption="Crash Detection.",use_container_width=False)
    with col8:
        st.title("Vehicle Crash Detection using Deep Learning.",)
        st.write("An Algorithmn developed using **LSTM AutoEncoder** Architecture to detect a low-high impact crash using vehicle telematics data.")
        st.markdown("""
        - Description:
            - 1.Performed end-end life cycle of an ML project like EDA,Feature Transformation, Ground Truth Labelling.
            - 2.Acheived a Decent Performace with test data & proved a POC.
        - skills used:
            - python
            - Deep Learning *(LSTM AutoEncoder)*
        - Tools used:
            - Jupyter Notebook
        - Frameworks used:
            - Pandas
            - Matplotlib,Seaborn,Plotly
            - Sklearn,keras,tensorflow

""",)

    st.divider()

    

    col9,col10 = st.columns([1,3])
        
    with col9:
        st.write("")
        st.write("")

        st.image(foodrecognization_image,caption="Food Recognition and Calorie Counting",use_container_width=False,width=200,)
    with col10:
        st.title("Food Recognition and Calorie Counting Using CNN",)
        st.write("An Algorithmn developed using **CNN** detect a food items from the image & using *NLP* to map calories.")
        st.markdown("""
        - Description:
            - an application that recognizes food images and calculates calorie content using Convolutional Neural Networks (CNN).
            - 88% accuracy in food recognition and 98% accuracy in database matching.
           
""",)

    
    st.divider()
    st.header("Get in touch with me!",anchor="msg")
    st.write(":heavy_minus_sign:" * (18))

    c41,c42 = st.columns(2)

    with c41:

        Name = st.text_input(label="Name",placeholder="Your Full Name")
        Email = st.text_input(label="Email",placeholder="Your Email Address")

        Message = st.text_area(label="Type Your Message",)

        btn = st.button(label="Send",)

        
        if btn:
            if Name:
                if Email:
                    if Message:
                        st.success("Received Your Message, will get back with u soon....")
                        new_row = pd.DataFrame({"Name": (Name), "Email": (Email), "Message":(Message), "Date&Time":(datetime.datetime.now())},index=[0])
                        df = pd.concat([logs, new_row], ignore_index=True)
                        df.to_csv("vishnu_portfolio_logs.csv",index=False)
                    else:
                        st.warning("Type Your Message")
                else:
                    st.warning("Enter your email.")
            else:
                st.warning("enter your name.")

        
    with c42:

        st_lottie(lottie_email,height=400,width=400)
    st.divider()


    

    st.header("Connect Me",anchor="contacts")
    st.write(":heavy_minus_sign:" * (10))

    c51,c52 = st.columns(2)

    with c51:

        st.write("- Gmail:-")
        st.markdown('<a href="mailto:vishnupuma18@gmail.com">vishnupuma18@gmail.com</a>', unsafe_allow_html=True)

        st.write("- Ping me at :-")
        st.markdown('<a href="tel:+91 8688353507">+91 8688353507</a>',unsafe_allow_html=True)

    with c52:

        file_ = open("./location.PNG","rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        
        st.write("Bangalore,India")
        st.markdown(f'''
            <a href='https://www.google.com/maps/place/Bengaluru,+Karnataka/@12.9539456,77.4661264,11z/data=!3m1!4b1!4m6!3m5!1s0x3bae1670c9b44e6d:0xf8dfc3e8517e4fe0!8m2!3d12.9715987!4d77.5945627!16zL20vMDljMTc?entry=ttu&g_ep=EgoyMDI0MTIwMi4wIKXMDSoASAFQAw%3D%3D'>
            <img src="data:image/gif;base64,{data_url}" width='300' height='200'/>
            </a>''',
            unsafe_allow_html=True
        )

    st.write("")
    st.write("")
    st.markdown("##### Follow Me on!")
    c53,c54,c55,c56 = st.columns(4)

    with c53:

        file_ = open("./linkedin.GIF","rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        
        st.write("Linked-In!")
        st.markdown(f'''
            <a href='https://www.linkedin.com/in/vishnu-sai-reddy-3a9b5a243/'>
            <img src="data:image/gif;base64,{data_url}" width='150' height='150'/>
            </a>''',
            unsafe_allow_html=True
        )

    with c54:

        file_ = open("./instagram.GIF","rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        
        st.write("Instagram!")
        st.markdown(f'''
            <a href='https://www.instagram.com/sai.basireddy/'>
            <img src="data:image/gif;base64,{data_url}" width='150' height='150'/>
            </a>''',
            unsafe_allow_html=True
        )

    with c55:
        file_ = open("./github.GIF","rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.write("Github!")
        st.markdown(f'''
            <a href='https://github.com/vishnureddysai'>
            <img src="data:image/gif;base64,{data_url}" width='150' height='150'/>
            </a>''',
            unsafe_allow_html=True
        )
    with c56:

        file_ = open("./mediumweb.GIF","rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.write("Medium!")
        st.markdown(f'''
            <a href='https://medium.com/@vishnupuma18'>
            <img src="data:image/gif;base64,{data_url}" width='150' height='150'/>
            </a>''',
            unsafe_allow_html=True
        )


