# Codeforces-recommender

此專案為 [YTP 少年圖靈計畫](https://www.tw-ytp.org/) 專題實作作品。

* 隊名: 好問題
* 作者: 臺中一中 蔡孟廷、田曜群、蔣承睿

## 簡介

作為世界上最知名的線上程式比賽網站之一，Codeforces 內有大量的題目。然而，部分題目時常被詬病出題品質不佳，例如數學能力需求遠大於程式競技能力需求，導致鑑別度低下，花費了時間卻不見得能夠提升競程能力。因此，作者們製作了一 Chrome 插件名為 Codeforces Recommender，主要目的在於向使用者推薦 Codeforces 題目，並使其能夠有效率的進步。

推薦題目的原理大致為，假設使用者解出了某道題目，並利用模型預測使用者解出該題目之後的 Rating 變化，若 Rating 增加越多，則表示題目越值得推薦。

模型方面，採用 Tensorflow GNN 為框架，以使用者和題目為點，通過關係為邊。經過採樣、分割資料集和訓練後，得到模型。預測時，在使用者和欲預測的題目間連邊，並預測使用者的點權。

專案主要分為兩部分，模型和推薦，前者為產生模型的過程及模型效能的驗證，後者則包含使用者介面、前後端互動，以及產生推薦列表。

## 流程

### 模型

資料蒐集->資料篩選->資料預處理->模型訓練->推薦驗證

### 推薦

使用者提出要求->篩選出題目列表->預測並排序->回傳推薦列表

## 模型程式碼檔案

### [fetch_data.py](https://github.com/CFRecommender/Codeforces-Recommender/blob/main/fetch_data.py) 


抓取 contest 內 AC 的 submission，紀錄使用者 Handle、Rating 以及題目的名字、難度和標籤 (以 bitmask 儲存)，並將每場比賽的資料各自存成一個 CSV 檔。

### [train.py](https://github.com/CFRecommender/Codeforces-Recommender/blob/main/train.py)
利用 contest 資料訓練模型，並儲存模型檔案和訓練過程折線圖。

### [show_rating_changes.py](https://github.com/CFRecommender/Codeforces-Recommender/blob/main/show_rating_changes.py)
計算一群通過特定題目的使用者到現在的 Rating 變化，以直方圖呈現變化分布

## 預測程式碼檔案

### [main.py](https://github.com/CFRecommender/Codeforces-Recommender/blob/main/main.py)
推薦主程式

### [filter_problem.py](https://github.com/CFRecommender/Codeforces-Recommender/blob/main/filter_problem.py)
依照使用者的要求過濾題目列表

### [sorting.py](https://github.com/CFRecommender/Codeforces-Recommender/blob/main/sorting.py)
預測並產生推薦順序。

向其中的 `sort_pb()` 傳入使用者 Handle 與題目列表，會利用模型預測該使用者通過列表中的題目後各自的 Rating 變化，並回傳推薦順序 (依預測結果排序)。
