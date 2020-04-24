#----------------------------------------------------Parse  XML-------------------------------------------------------#
#---------------------------------------------------April 2020---------------------------------------------------------#

# Import packages
import xml.etree.cElementTree as et
import pandas as pd
import os
import re

# Function to remove html tags from a string
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Function to remove multiple spaces
def remove_spaces(text):
    text=re.sub(' +',' ',text).strip()
    return text

# Function to parse XML
def import_xml(file):
    xmlp = et.XMLParser(encoding="UTF-8")
    parsed_xml = et.parse(file,parser=xmlp)
    root = parsed_xml.getroot()
    for child in root.findall('Obj'):
        title=child.find('TitleAtt').find('Title').text
        type=child.find('ObjectTypes').find('mstar').text
        startdate=child.find('StartDate').text
        enddate=child.find('EndDate').text

        if child.find('Abstract') != None:
            for node in child.iter('AbsText'):
                abstract=node.text
                abstract=remove_spaces(remove_html_tags(abstract))
        else:
            abstract=''

    if root.find('TextInfo') != None:
        for node in root.iter('Text'):
            text=node.text
            text=remove_spaces(remove_html_tags(text))
            wordcount=node.get('WordCount')
    else:
        text=''
        wordcount=0

    for child in root.findall('DFS'):
        pubtitle=child.find('PubFrosting').find('Title').text
    new_dic={'Title':title,'Type':type,'StartDate':startdate,'EndDate':enddate,'Abstract':abstract,'Text':text,
        'TextWordCount':wordcount,'PubbTitle':pubtitle}
    df_xml=pd.DataFrame([new_dic])
    return df_xml

# Parse XML files
filePath='Data/TDM Studio/Sample XML/'
df=pd.DataFrame()
for filename in os.listdir(filePath):
    if filename.endswith('.txt'):
        file=filePath+filename
        new=import_xml(file)
        df=df.append(import_xml(file)).reset_index(drop=True)
df.to_pickle('Data/TDM Studio/Sample XML/parsed_xml.pkl')
