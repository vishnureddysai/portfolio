import google.generativeai as genai
import PyPDF2 as pdf


genai.configure(api_key="AIzaSyCsfmhNSqygZjwDt01-K3ZjKaSAmNKRWmE")
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_llm_answer(questions):
    with open("./Resume vishnu masters final.pdf", "rb") as file:
    
        reader = pdf.PdfReader(file)
        # print(len(reader.pages))

        text = ""
        for page in range(len(reader.pages)):
            page = reader.pages[page]
            text+=str(page.extract_text())
        
    input_prompt = """
    Hey you are personal assitant to vishnu where you know more details about him basically  you know about his portfolio like his education, carrer and acheivements etc.
    vishnu context: {text}
    answer based on the user question: {question}
    """.format(text=text,question=questions)

    response = model.generate_content(input_prompt)
    return (response.text)
