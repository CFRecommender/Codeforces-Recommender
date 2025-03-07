import requests
import matplotlib.pyplot as plt
import numpy as np

seen = set()
seen_current = set()

def get_contest_submissions(contest_id):
    url = f"https://codeforces.com/api/contest.status?contestId={contest_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data["status"] == "OK":
            print("fetched submissions")
            return data["result"]
        else:
            print("Error in API response:", data.get("comment", "Unknown error"))
            return []
    except requests.exceptions.RequestException as e:
        print("An error occurred while fetching submissions:", e)
        return []

def get_users_solved_problem(contest_id, problem_name):
    submissions = get_contest_submissions(contest_id)
    global seen
    seen2 = set()

    if submissions:
        for submission in submissions:
            if submission.get("verdict") == "OK" and submission["problem"]["name"] == problem_name:
                user = submission["author"]["members"][0]["handle"]
                if user not in seen2:
                    seen2.add(user)
        if not seen:
            seen = seen2
        else:
            seen = seen & seen2

def get_contest_ratings(contest_id):
    url = f"https://codeforces.com/api/contest.ratingChanges?contestId={contest_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        ret = {}

        if data["status"] == "OK":
            print("fetched rating changes")
            for change in data["result"]:
                if change["handle"] in seen and change["oldRating"] <= 1400:
                    seen_current.add(change["handle"])
                    ret[change["handle"]] = change["oldRating"]
            return ret
        else:
            print("Error in API response:", data.get("comment", "Unknown error"))
            return {}
    except requests.exceptions.RequestException as e:
        print("An error occurred while fetching rating changes:", e)
        return {}

def get_user_current_ratings(users):
    user_ratings = {}
    batch_size = 100
    for i in range(0, len(users), batch_size):
        user_batch = users[i:i + batch_size]
        handles = ";".join(user_batch)
        url = f"https://codeforces.com/api/user.info?handles={handles}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data["status"] == "OK":
                for user_data in data["result"]:
                    handle = user_data["handle"]
                    rating = user_data.get("rating", 0)
                    user_ratings[handle] = rating
            else:
                print("Error in API response:", data.get("comment", "Unknown error"))

        except requests.exceptions.RequestException as e:
            print("An error occurred while fetching user ratings:", e)

    return user_ratings

def plot_rating_changes(rating_changes):
    if not rating_changes:
        print("No rating data available for plotting.")
        return
    
    min_rating = -500
    max_rating = 2000
    bins = np.arange(min_rating, max_rating + 1, 50)
    
    counts, bin_edges = np.histogram(rating_changes, bins=bins)
    total_users = sum(counts)
    proportions = counts / total_users * 100

    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], proportions, width=50, color="skyblue", edgecolor="black", alpha=0.7, align="edge")

    xticks = np.arange(min_rating, max_rating + 1, 100)
    yticks = np.arange(0,12+1,2)
    plt.xticks(xticks, rotation=45)
    plt.yticks(yticks)
    plt.xlabel("Rating Changes")
    plt.ylabel("Proportion of Users(%)")
    plt.title("Rating Changes Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.show()


contest_id = [1691,1695,1725,1761,1766]
problem_name = ["Shoe Shuffling","Circle Game","Basketball Together","Elimination of a Ring","Notepad#"]

for i in range(len(contest_id)):
    get_users_solved_problem(contest_id[i], problem_name[i])

old_ratings = get_contest_ratings(contest_id[0])
current_ratings = get_user_current_ratings(list(seen_current))
rating_changes = []
for handle in old_ratings:
    if handle in current_ratings:
        rating_changes.append(current_ratings[handle] - old_ratings[handle])

if rating_changes:
    plot_rating_changes(rating_changes)
else:
    print("No users solved this problem in the contest.")

print(len(seen_current))
print(np.mean(rating_changes))
