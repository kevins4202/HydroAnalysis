import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import re
import requests
from random import randint

#Import dataframe of all 26,000 papers and all keywords/buzzwords
papers = pd.read_csv(os.getcwd().replace('scripts', 'data/papers') + '/final_valid_papers.csv')
kwords = pd.read_csv(os.getcwd().replace('scripts', 'data')+'/technical_buzzwords.csv')

# make the dataframe store words as a list instead of long string
words = pd.DataFrame(columns = ['topic', 'keywords'])
for i in range(kwords.shape[0]):
    words.loc[len(words.index)] = [str(kwords.iat[int(i),0]), str (kwords.iat[int(i),1]).split(", ")]

# print(kwords)

#API KEYS
keys = [
    "d0c39572282257204e73e16476da7d80",
    "9cd9459f5f9679f35f791efa0146f3f7",
    "4fed3a6ea82674610dc37e597f4c1b1c",
    "d6808e44f5904f1fd9a01e0404f56bc1",
    "146b970d412eaae2851db7c51d2a731b",
    "48e96520d3ddbb4c4abc45ad6b0421d6",
    "4e23a40cfe1ab144604586aae4c42565",
    "4b32aea068998554f30a282f965c271e",
    "a1e7378d4d1046a8fa9bbf1750d480a4"
]

#find keywords in paper full text
def find_keywords(k_list, abstract):
    ret = [] # all instances of found buzzwords
    
    old_abstract = abstract
    
    orig = old_abstract
    
    #loop thru all buzzwords
    for word in k_list:
        word = word.lower().replace('-', ' ')
        curr_abstract = old_abstract
        c_orig = orig
        done = False
        cnt = 0
        while(True):
            if done:
                break
            try:
                #try to find word
                found_i = curr_abstract.index(word)

                #find context of 100 characters around it
                l_b = max(0, found_i - 100)
                u_b = min(len(curr_abstract), found_i+len(word)+100)
                
                if(l_b > u_b):
                    break
                else:
                    cnt+=1

                if(u_b == len(curr_abstract)):
                    break

                #shorten the text to take out the found keyword
                
                curr_abstract = curr_abstract[found_i + len(word):]
                c_orig = c_orig[found_i + len(word):]

            except ValueError:
                # print("DONE")
                #if cannot find then we are done
                done = True
                continue
            except Exception as e:
                break
        if cnt != 0:
            #append number of topics found and the specific keyword
            ret.append([word,cnt])

            # print([word, cnt])
    
    return ret#returns the specific word & context

def cleaning(tmp): #PREPROCESS FULL TEXT
    #     print(tmp)
    #remove non-ascii
#     tmp = unicodedata.normalize('NFKD', tmp).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    #remove URLs
#     tmp = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', str)
    #remove punctuations
    tmp = re.sub(r'[^\w]|_',' ',tmp)
    #to lowercase
    tmp = tmp.lower()
    #Remove additional white spaces
    tmp = re.sub('[\s]+', ' ', tmp)

    return tmp
       
keyi = 0 #API KEY INDEX
# INSTANCES OF FOUND BUZZWORDS
buzzwords_found = pd.DataFrame(columns = ["index", "doi", "buzz_id","category", "subword", "count"])
#CONTEXT OF KEYWORDS (NOT USED)
contexts = pd.DataFrame(columns = ["index", "doi", "context"])
#ERROR PAPERS
error_df = pd.DataFrame(columns = ["index", "doi"])
# CORPUS OF FULL TEXTS
full_text_df = pd.DataFrame(columns = ["index", "text"])

#get the full text
def fullText(i, keyi):
    #doi of paper at this index
    doi = papers.iat[i,4]
#     done = False
#     print(doi)
    try:
        #make the request
        js = requests.get(
            f"https://api.elsevier.com/content/article/doi/{doi}",
            headers = {"X-ELS-APIKey":keys[keyi], "Accept":"application/json"}
,
        )
        
        r = js.json()

        full_text = r['full-text-retrieval-response']['originalText']
#         full_text = re.sub(r'^https?:\/\/.[^\s]+', '', full_text)
#         print(full_text)
        
        #GET THE ABSTRACT
        abstr = r['full-text-retrieval-response']['coredata']['dc:description'][:25]
         
        index = full_text.find(abstr)
        if(index==-1):
            #BROKEN
            other.append(i)
            return None
        

        #CUT OFF FULL TEXT SO IT STARTS FROM ABSTRACT
        full_text = full_text[index:]
        
        #CUT OFF FULL TEXT SO IT ENDS BEFORE THE REFERENCES
        if full_text.find("References") == -1:
            other.append(i)
            return None
        full_text = ''.join(full_text.split('References')[:-1])

        #PREPROCESS THE WHOLE TEXT
        full_text = cleaning(full_text)    
#         print(full_text)
        
        #INCREMENT THE KEY INDEX
        keyi+=1
        
        return full_text
    except Exception as e:
        if js.status_code == 404:
            error_df.loc[len(error_df.index)] = [i,doi]
            print("NOT FOUND")
        elif js.status_code ==429:
            print("KEY ERROR")
            keyi+=1
            return fullText(i, (keyi)%9)
        else:
            print("OTHER ERROR")
            error_df.loc[len(error_df.index)] = [i,doi]
        return None

#RUN!!!!!!!!!!!!!!!!!!!  
error_df_i = 0
for i in range(papers.shape[0]):
    print(i)
    if i%1500==0 and i !=0:
        full_text_df.to_csv('/Volumes/mac/corpus/text_corpus'+str(error_df_i)+'.csv', index = False)
        error_df_i+=1
        full_text_df = pd.DataFrame(columns = ["index", "text"])

    # print(i)
    ft = fullText(i = i, keyi = keyi)
    
    if ft == None:
        error_df.loc[len(error_df.index)] = [i,papers.iat[i,4]]
    else:
        full_text_df.loc[len(full_text_df.index)] = [i, ft]
#     print(ft)
    for j in range(words.shape[0]):
        subterms = words.iat[j, 1]
#         print(subterms)
        for found in find_keywords(subterms, ft):
            buzzwords_found.loc[len(buzzwords_found.index)] = [i,papers.iat[i,4], j,words.iat[j,0], found[0], found[1]]

buzzwords_found.to_csv('found_buzzwords.csv', index = False)
error_df.to_csv('error_full_text.csv', index = False)
full_text_df.to_csv('/Volumes/mac/corpus/text_corpus'+str(error_df_i+1)+'.csv', index = False)