from llama_index.multi_modal_llms.ollama import OllamaMultiModal
mm_model = OllamaMultiModal(model="phi3")
tab ="""  Degree/Certificate                            Institute/Board  CGPA/Percentage          Year  
0        B.Tech. CSE  Indian Institute of Technology, Hyderabad     9.65  2022-Present 
1   Senior Secondary                Noble Kingdom Public School    96.2%          2022  
2          Secondary                Noble Kingdom Public School    96.83%          2020 
"""
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="""You are a table summarizer, which will be given table data.Each row of the table(including header) is given.
        You are supposed to first print all the column names of the table.  Then, print the number of rows. then summarize each row into 1 line meaningful sentence. One sentence per row.
        Include all columns in summary of each row. Do not include any extra information. Do not include any notes or comments. 
        Produce the row-wise summary only. Make each row summary like suppose table has columns restaurant name, location, rating, then summary of each row should be like The restaurant name is located in location and has rating stars"""
    ),
    ChatMessage(role="user", content= tab),
]

resp = mm_model.chat(messages)
print(resp)