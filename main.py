from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import filter_problem as fp

fp.get_problem_url()
import sorting
fp.load_all_csv()
all_tags = ['*special', '2-sat', 'binary search', 'bitmasks', 'brute force', 'chinese remainder theorem', 'combinatorics', 'constructive algorithms', 'data structures', 'dfs and similar', 'divide and conquer', 'dp', 'dsu', 'expression parsing', 'fft', 'flows', 'games', 'geometry', 'graph matchings', 'graphs', 'greedy', 'hashing', 'implementation', 'interactive', 'math', 'matrices', 'meet-in-the-middle', 'number theory', 'probabilities', 'schedules', 'shortest paths', 'sortings', 'string suffix structures', 'strings', 'ternary search', 'trees', 'two pointers']
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://172.20.10.10:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],  # Allow all origins (or use `origins` list for security)
    allow_credentials = True,
    allow_methods = ["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers = ["*"],  # Allow all headers
)

@app.get("/")
async def read_root():
    return {"hint": "/{user_id}?difficulty/tag"}


@app.get("/{user_id}")
async def read_user(user_id: str, difficulty: str, tag: Optional[str] = None):
    if user_id not in fp.user_dict:
        return {"error": "Handle Not Found"}
    tag_list = None
    if tag != "all":
        tag_list = list("0000000000000000000000000000000000000")
        try:
            idx = all_tags.index(tag)
            tag_list[idx] = "1"
        except ValueError:
            return {"error": "Invalid tag format"}
    user_rating = fp.user_dict[user_id]["rating"]
    if difficulty == "Easy":
        user_rating -= 200
    elif difficulty == "Hard":
        user_rating += 200
    min_rating = user_rating - 100
    max_rating = user_rating + 100
    filtered = fp.filter_problems(user_id, min_rating = min_rating, max_rating = max_rating, required_tags = tag_list)
    tmp = list(filtered.keys())
    ret_list = sorting.sort_pb(user_id, tmp)[:5]
    ret = []
    for i in ret_list:
        ret.append(filtered[i])
    return {"user_id": user_id, "problems_list": ret}
