import requests
import time
import pandas as pd

def get_all_problems():
    url = "https://codeforces.com/api/problemset.problems"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] != "OK":
            print("Error in API response:", data.get("comment", "Unknown error"))
            return None
        
        problems = data["result"]["problems"]
        all_tags = sorted({tag for problem in problems if "tags" in problem for tag in problem["tags"]})
        
        problem_dict = {}
        
        for problem in problems:
            if "name" in problem and "rating" in problem:
                tag_vector = "".join(["1" if tag in problem.get("tags", []) else "0" for tag in all_tags])
                
                problem_dict[problem["name"]] = {
                    "rating": problem["rating"],
                    "tags": tag_vector
                }
        print("Fetched whole problemset")
        return problem_dict
    
    except requests.exceptions.RequestException as e:
        print("An error occurred while fetching problem data:", e)
        return None

def get_all_contests():
    url = "https://codeforces.com/api/contest.list?gym=false"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data["status"] == "OK":
            contests = data["result"]
            contest_ids = []

            for contest in contests:
                if contest["phase"] != "BEFORE":
                    contest_ids.append(contest["id"])
            print("Fetched all contests")
            return contest_ids
        else:
            print("Error in API response:", data.get("comment", "Unknown error"))
            return []
    except requests.exceptions.RequestException as e:
        print("An error occurred while fetching contest data:", e)
        return []
    
def get_contest_submissions(contest_id):
    url = f"https://codeforces.com/api/contest.status?contestId={contest_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        if data["status"] == "OK":
            return data["result"]
        else:
            print("Error in API response:", data.get("comment", "Unknown error"))
            return []
    except requests.exceptions.RequestException as e:
        print("An error occurred while fetching submissions:", e)
        return []

def get_contest_ratings(contest_id):
    url = f"https://codeforces.com/api/contest.ratingChanges?contestId={contest_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "OK":
            return {entry["handle"]: entry["newRating"] for entry in data["result"]}
        else:
            print("Error in API response:", data.get("comment", "Unknown error"))
            return {}
    except requests.exceptions.RequestException as e:
        print("An error occurred while fetching contest ratings:", e)
        return {}

def save_to_csv(contest_id, problem_dict):
    datas = []
    seen = {}
    submissions = get_contest_submissions(contest_id)

    if submissions:
        print(f"Fetched {len(submissions)} submissions")
        for submission in submissions:
            if 'verdict' in submission and submission['verdict'] == 'OK' :
                user = submission["author"]["members"][0]["handle"]
                problem = submission["problem"]["name"]
                if (user,problem) not in seen:
                    seen[(user,problem)] = 1
                    if problem not in problem_dict or user not in user_dict:
                        continue
                    difficulty = problem_dict[problem]["rating"]
                    tags = problem_dict[problem]["tags"]
                    datas.append({
                        "User": user,
                        "Rating": user_dict[user],
                        "Problem Name": problem,
                        "Difficulty": difficulty,
                        "tags": tags
                    })
                    
        filename = f"{contest_id}.csv"
        df = pd.DataFrame(datas)
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"Saved {len(datas)} accepted submissions to {filename}")
    else:
        print("No submissions fetched.")


if __name__ == "__main__":
    count = 10
    problem_dict = get_all_problems()
    contest_ids = get_all_contests()[:count]
    user_dict = {}
    for i in contest_ids:
        start_time = time.time()
        tmp = get_contest_ratings(i)
        if len(tmp) == 0:
            continue
        for key in tmp:
            if key not in user_dict:
                user_dict[key] = tmp[key]
        save_to_csv(i, problem_dict)
        print(f"Execution time: {time.time() - start_time:.2f} seconds")