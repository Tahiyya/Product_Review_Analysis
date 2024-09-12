import openai
import time
import pandas as pd

MODEL_NAME = "phi3:mini"

print("Client setup")
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="nokeyneeded",
)

CONS_SYSTEM_MSG = "Raw insights from customer reviews on the cons of iPhone 13 are given. Clean and format the data properly, and rephrase the data to generate one single consolidated list of cons. Return the response in the form of a single unordered list only."
input_file2 = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\4Final_cons_list.csv"

# df = pd.read_csv(input_file1)
df2 = pd.read_csv(input_file2)

txt2 = ""
for i in df2["pros_cons"]:
    i = i.replace("\n","")
    txt2 = txt2 + i + " "

response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.2,
    n=1,
    messages=[
        {"role": "system", "content": CONS_SYSTEM_MSG},
        {"role": "user", "content": txt2},
    ],
)
response_list2 = response.choices[0].message.content
print("Response:")
print(response_list2)
# print(response.choices[0].text)

# Save the resulting list of pros and cons to a new Excel file for further offline processing
df_proscons2 = pd.DataFrame()
df_proscons2["cons"] = [response_list2]
output_file_proscons2 = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\5Final_cons_listed.csv"
df_proscons2.to_csv(output_file_proscons2, index=False)

PROS_SYSTEM_MSG = "Raw insights from customer reviews on the pros of iPhone 13 are given. Clean and format the data properly, and rephrase the data to generate one single consolidated list of pros. Return the response in the form of a single unordered list only."
input_file1 = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\4Final_pros_list.csv"

# df = pd.read_csv(input_file1)
df1 = pd.read_csv(input_file1)

txt1 = ""
for j in df1["pros_cons"]:
    j = j.replace("\n","")
    txt1 = txt1 + j + " "

response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.2,
    n=1,
    messages=[
        {"role": "system", "content": PROS_SYSTEM_MSG},
        {"role": "user", "content": txt1},
    ],
)
response_list1 = response.choices[0].message.content
print("Response:")
print(response_list1)
# print(response.choices[0].text)

# Save the resulting list of pros and cons to a new Excel file for further offline processing
df_proscons = pd.DataFrame()
df_proscons["pros"] = [response_list1]
output_file_proscons = r"C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\5Final_pros_listed.csv"
df_proscons.to_csv(output_file_proscons, index=False)