# Sentiment Analysis
Description
---
透過PYPPT修正檔案，取得PTT Stock盤中堆文 

使用安裝
---
- step01
資料準備
```
git clone https://github.com/lzrong0203/PTT_Stock_sentiment.git
pip install -r requirements.txt
git clone -b "stock_market_time" https://github.com/lzrong0203/PyPtt.git NewPyPtt
```

- step02
帳號設置(.env)
```
OPENAI_API_KEY = sk-...  
OPENAI_ORGANIZATION = org-...   
PTT_ID = ...  
PTT_PW = ...  
````

PyPtt_generate_stock_posts_dataset.py
---
###import方式
###同層NewPyPtt資料夾內容
```
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'NewPyPtt'))
from dotenv import load_dotenv
from PyPtt import PTT as PyPtt
from PyPtt import data_type
from PyPtt import screens
```

###確認環境參數設定
###post_num設定抓取推文數
```
if __name__ == '__main__':
    # 將id與pw存在.env中，透過此指令將環境變數引入
    load_dotenv()
    # 建立物件
    pyptt_obj = PyPttObj()
    # 登入
    pyptt_obj.login(os.getenv('PTT_ID'), os.getenv('PTT_PW'))
    # 建立資料集
    pyptt_obj.generate_newest_posts_dataset(
        post_num=2, save_path='ptt_stock_posts_num1.csv')
    # 登出
    pyptt_obj.logout()
```

### 結果資料
- 資料格式  
![image]
- Pushes欄位中的資料  
![image]
- 資料集可見 sentiment_analysis/dataset/ptt_stock_posts_num1.csv 

