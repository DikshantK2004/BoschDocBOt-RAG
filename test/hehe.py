import os

# Define the mappings of (cid:xx) to characters
mappings = {
    "(cid:36)": "A",
    "(cid:37)": "B",
    "(cid:38)": "C",
    "(cid:39)": "D",
    "(cid:40)": "E",
    "(cid:41)": "F",
    "(cid:42)": "G",
    "(cid:43)": "H",
    "(cid:44)": "I",
    "(cid:45)": "J",
    "(cid:46)": "K",
    "(cid:47)": "L",
    "(cid:48)": "M",
    "(cid:49)": "N",
    "(cid:50)": "O",
    "(cid:51)": "P",
    "(cid:52)": "Q",
    "(cid:53)": "R",
    "(cid:54)": "S",
    "(cid:55)": "T",
    "(cid:56)": "U",
    "(cid:57)": "V",
    "(cid:58)": "W",
    "(cid:59)": "X",
    "(cid:60)": "Y",
    "(cid:61)": "Z",
    "(cid:68)": "a",
    "(cid:69)": "b",
    "(cid:70)": "c",
    "(cid:71)": "d",
    "(cid:72)": "e",
    "(cid:73)": "f",
    "(cid:74)": "g",
    "(cid:75)": "h",
    "(cid:76)": "i",
    "(cid:77)": "j",
    "(cid:78)": "k",
    "(cid:79)": "l",
    "(cid:80)": "m",
    "(cid:81)": "n",
    "(cid:82)": "o",
    "(cid:83)": "p",
    "(cid:84)": "q",
    "(cid:85)": "r",
    "(cid:86)": "s",
    "(cid:87)": "t",
    "(cid:88)": "u",
    "(cid:89)": "v",
    "(cid:90)": "w",
    "(cid:91)": "x",
    "(cid:92)": "y",
    "(cid:93)": "z",
    "(cid:483)": "(",
    "(cid:484)": ")",
    "(cid:3)": " ",
}

# Function to replace (cid:xx) with characters based on mappings
def replace_cid(content):
    for cid, char in mappings.items():
        content = content.replace(cid, char)
    return content

# Path to the folder containing TXT files
folder_path = "/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/documents_aditya/exter.pdf"

# Iterate over all TXT files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        # Read the content of the file
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        # Replace (cid:xx) with characters
        modified_content = replace_cid(content)
        # Write the modified content back to the file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(modified_content)
