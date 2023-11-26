import sys
#print(sys.path)
sys.path.append('.')
import PyPDF2
import requests
import random
import json
from hashlib import md5
import argparse
import os
import time

#####################################################################################################################################################################
#    
#   PDF 用百度翻译 输出到 pdfName_百度翻译.md  和 pdfName_rawtxt.md
#   -- 缺省不调用baidu翻译，只输出pdf rawtxt
#   -- 推荐使用脚本 pdf_GPT_translatPolish.sh  联动调用 pdf_translator 提取文本， doubleTranslator.py 进行GPT 直译、意译, 效果好比百度直译好，GPT还能自动调整格式。
#   -- Deprecated:  脚本 pdf_baidu_translatGPTPolish.sh 联动调用 baidu翻译和 chiesePolish.py 进行GPT 二次意译
#
######################################################################################################################################################################

# Set up your Baidu API credentials
appid = '20230407001631731'
appkey = '2zoVP_v_5yF0jMRZiH9P'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'en'
to_lang =  'zh'
endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

print("\n############# -- 推荐使用脚本 pdf_GPT_translatPolish.sh  联动调用 doubleTranslator.py 进行GPT 直译、意译, 效果好比百度直译好，GPT还能自动调整格式。\n############# -- 注意本程序是文件添加模式，运行前先清除备份原翻译和rawtext文件\n############# -- 缺省不翻译只提取rawtext，请用 --do_translate 调用baidu翻译")
parser = argparse.ArgumentParser(description='Translate PDF file using Baidu API')
parser.add_argument('pdf_file', type=str, help='Path to PDF file')
parser.add_argument('--do_translate', action='store_true',default=False,  help='Only extract raw text by default')
# Parse arguments
args = parser.parse_args()

do_translate=args.do_translate
# Get the input file name without the '.pdf' extension
input_file_name = os.path.splitext(args.pdf_file)[0]

# Set the output file name to the input file name with a '.txt' extension
output_file_name = input_file_name + '_百度翻译.md'
rawtxt_output_file_name=input_file_name + '_rawtxt.md'
if do_translate:
    print( "\n##### 百度翻译 输出文件 ："+output_file_name)
print( "\n##### 文本提取 输出文件 ："+rawtxt_output_file_name)

# Open the PDF file in read-binary mode
with open(args.pdf_file, 'rb') as f: 
    # Create a PDF reader object
    reader = PyPDF2.PdfReader(f)

    # Get the number of pages in the PDF file
    num_pages =len(reader.pages)

    # Loop through each page and extract the text
    for i in range(num_pages): 
        pageNum=i+1   
        pageNum_str=str(pageNum)   
        # Get the page object
        page = reader.pages[i]
        # Extract the text from the page
        text = page.extract_text()

        num_newlines = text.count('\n')   
        # 有些PDF 里图片文本窗口里存在大量的短行，把换行后添加空格，增加原文可读性
        text = text.replace('\n', '\n ')

        # Page filter continue the loop  ,
        # debug only, remove after debug over       
        # if  pageNum>3:  # for example ,  if pageNum==1   or  pageNum>3 :
        #     print("\n ###### pass page ",pageNum);
        #     continue 
        
        print("\n\n ####### page ",pageNum, '  raw text  :\n', text )
        
        if do_translate :
            # Build baidu translator request
            salt = random.randint(32768, 65536)
            sign = make_md5(appid + text + str(salt) + appkey) 
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            payload = {'appid': appid, 'q': text, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
            # Add error handler for request post
            try:
                r = requests.post(url, params=payload, headers=headers)
                r.raise_for_status()
            except requests.exceptions.HTTPError as errh:
                print("HTTP Error:", errh)
            except requests.exceptions.ConnectionError as errc:
                print("Error Connecting:", errc)
            except requests.exceptions.Timeout as errt:
                print("Timeout Error:", errt)
            except requests.exceptions.RequestException as err:
                print("Something went wrong with the request:", err)

            result = r.json()
  
            # Write the translated text to the output file
            with open(output_file_name, 'a') as f:
                # Check if the 'trans_result' key exists in the 'result' dictionary
                if 'trans_result' in result:
                    f.write("##### page "+ pageNum_str+" : \n" )
                    # If it exists, loop through each translation and write it to the output file
                    for translation in result['trans_result']:
                        translated_text = translation['dst']
                        f.write(translated_text)
                    f.write('\n\n')
                    print("\n ####### page ",pageNum, " translated text(部分) :\n", translated_text, "\n");
                   
                else:
                    # If it doesn't exist, print an error message
                    print("\n ###### Page ", pageNum," error : 'trans_result' key not found in response from Baidu API")
                    f.write("\n##### page "+pageNum_str+" : translate fail.\n\n" )
                     

        with open(rawtxt_output_file_name, 'a') as rawtxt_file:
            rawtxt="\n\n##### page "+ pageNum_str+" : \n\n"+text  
            rawtxt_file.write(rawtxt ) 

        # Add a delay for 0.3s
        time.sleep(0.3)

                
        #end of program


        
         