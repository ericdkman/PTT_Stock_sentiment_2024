import time
import pandas as pd
import openai
import json
from dotenv import load_dotenv
import tiktoken

# openai 的 key 存放在同個資料夾中的 .env，需先透過此指令將環境變數引入
load_dotenv()

class GPTStockChatAnalyzer:
    def __init__(self):
        # 初始化 OpenAI
        openai.organization = "org-"
        openai.api_key = "sk-"
        self.token_sum = 0

    def get_completion(self, prompt, model="gpt-4o"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,  # this is the degree of randomness of the model's output
        )
        self.token_sum += response['usage']['total_tokens']
        return response.choices[0].message["content"], response["usage"]

    def sentiment_prompt(self, stock_market_chat):
        return f"""
        
        1. 請分析以下股市討論的情緒，並以「使用者的角度」去「仔細」分析且必須回答該文情緒分析的結果是 "positive"、"negative" 或 "neutral"。

        2. Your answer is a single word, either "positive" or "negative" or "neutral". 
           如果討論中沒有提及股票代碼或代號，請辨認情緒為 "neutral"。
           否則，請將該討論指涉的股票代碼或代號列在答案中。如果沒有提及股票代碼或代號，請將答案設為空字串 ""，並儘量分析文章隱含的情感。

        3. Provide above answers in just JSON format which cloud convert to Pandas DataFrame directly with keys for the series chats separated by angle brackets, respectively:
           Sentiment, Stocktarget.
         Follow the format:
        
        {{
            {{"Sentiment": ..., "Stocktarget": ...}}
           
            
        }}
        
        Chat text: {stock_market_chat}

        請注意，請基於「文本的整體意思和情感」來回答，而不僅僅是依賴特定詞語。
        Please note, it is essential to adhere to the rules specified by the JSON formats and values.

        """

    def analyze_sentiment(self, stock_market_chat):
        prompt = self.sentiment_prompt(stock_market_chat)
        try:
            response, token_usage = self.get_completion(prompt)
        except Exception as e:
            print(e)
            return None
        return response

def read_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def analyze_articles(df):
    # 建立 GPTStockChatAnalyzer 物件
    analyzer = GPTStockChatAnalyzer()
    sentiment_results = []

    for index, row in df.iterrows():
        # 取得每篇文章的內容
        article_content = row['Content']
        
        # 進行情緒分析
        sentiment_analysis_result = analyzer.analyze_sentiment(article_content)
        
        if sentiment_analysis_result is not None:
            sentiment_results.append(sentiment_analysis_result)
        else:
            sentiment_results.append("情緒分析失敗，請檢查程式設定及環境。")
        
        # 限制每分析一篇文章後等待 1 秒，以避免 OpenAI API 頻率限制
        time.sleep(1)
    
    return sentiment_results, analyzer.token_sum

if __name__ == '__main__':
    
    # 讀取包含許多文章的 CSV 檔案
    csv_file = '//Users/88696/PTT_Stock_sentiment/sentiment_analysis/ptt_stock_posts_test.csv'
    df = read_csv_data(csv_file)

    # 進行文章的情緒分析
    sentiment_results = analyze_articles(df)

    # 將情緒分析結果加入 DataFrame
    df['Sentiment_Result'] = sentiment_results

    # 儲存包含情緒分析結果的 DataFrame 到新的 CSV 檔案
    df.to_csv('//Users/88696/PTT_Stock_sentiment/sentiment_analysis/ptt_stock_posts_test.csv', index=False, encoding='utf_8_sig')
    
    print('done')
    #print('tokens:', analyzer.token_sum)
    
