import os
import pandas as pd
import requests
import csv

problem_url = {}
problem_dict = {}
user_dict = {}
all_tags = ['*special', '2-sat', 'binary search', 'bitmasks', 'brute force', 'chinese remainder theorem', 'combinatorics', 'constructive algorithms', 'data structures', 'dfs and similar', 'divide and conquer', 'dp', 'dsu', 'expression parsing', 'fft', 'flows', 'games', 'geometry', 'graph matchings', 'graphs', 'greedy', 'hashing', 'implementation', 'interactive', 'math', 'matrices', 'meet-in-the-middle', 'number theory', 'probabilities', 'schedules', 'shortest paths', 'sortings', 'string suffix structures', 'strings', 'ternary search', 'trees', 'two pointers']

def get_problem_url():
    print("getting urls...")
    url = "https://codeforces.com/api/problemset.problems"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] != "OK":
            print("Error in API response:", data.get("comment", "Unknown error"))
            return None
        
        problems = data["result"]["problems"]
        
        for problem in problems:
            contest_id = problem["contestId"]
            index = problem["index"]
            problem_url[problem["name"]] = f"https://codeforces.com/contest/{contest_id}/problem/{index}"
        return problem_url
    
    except requests.exceptions.RequestException as e:
        print("An error occurred while fetching problem data:", e)
        return None

def load_all_csv():
    print("loading files...")
    folder_path = "./Datas"
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            submissions = df.values.tolist()
            
            for user,rating,problem,difficulty,tags in submissions:
                if user not in user_dict:
                    user_dict[user] = {
                        "rating": rating,
                        "solved": {}
                    }
                if problem not in user_dict[user]["solved"]:
                    user_dict[user]["solved"][problem] = 1
                problem_dict[problem] = {
                    "rating": difficulty,
                    "tags": tags,
                }
            
        except Exception as e:
            print(f"無法讀取 {file}: {e}")
    print("loaded all files")

def filter_problems(user, min_rating=None, max_rating=None, required_tags=None):
    filtered_problems = {}
    try:
        for problem in problem_dict:
            problem_rating = problem_dict[problem]["rating"]
            problem_tags = problem_dict[problem]["tags"]

            if min_rating is not None and problem_rating < min_rating:
                continue
            if max_rating is not None and problem_rating > max_rating:
                continue
            
            flag = 1
            if required_tags != None:
                flag = 0
                for i in range(37):
                    if required_tags[i] == '1' and problem_tags[i] == '1':
                        flag = 1
            if problem in user_dict[user]["solved"]:
                flag = 0
            if flag:
                filtered_problems[problem] = {
                    "name": problem,
                    "url": problem_url[problem],
                    "rating": problem_rating
                }
    except Exception as e:
        print(f"無法讀取 {user}: {e}")
    
    return filtered_problems
