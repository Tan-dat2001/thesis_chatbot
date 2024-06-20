import random
import json
import torch
import mysql.connector
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import re

#api
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://thesis-chatbot-a8gh.onrender.com",
    "http://127.0.0.1:5500/index.html",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8000"
]
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Cho phép tất cả các nguồn gốc (bạn có thể giới hạn theo nhu cầu)
    allow_credentials=True,
    allow_methods=["GET","POST"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các tiêu đề
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
# print(f"all world: {all_words}")

def query_database(bedrooms=None, bedrooms_condition=None, floors=None, floors_condition=None, toilets=None, toilets_condition=None, area=None, area_condition=None,price=None, price_condition=None):
    # Cập nhật thông tin kết nối
    conn = mysql.connector.connect(
        host="13.212.22.175",   # Địa chỉ host của database đã deploy
        user="house_sale_admin",        # Tên đăng nhập
        password="HouseSale2024!",    # Mật khẩu
        database="house_sale"       # Tên cơ sở dữ liệu
    )
    cursor = conn.cursor()
    base_query = "SELECT propertyId FROM Properties WHERE 1=1 AND status='Available'"
    
    params = []
    
    print(f"params query: {price} {price_condition}")
    if bedrooms is not None and bedrooms_condition is not None:
        if bedrooms_condition == "less_than":
            base_query += " AND numberOfBedRoom <= %s"
        elif bedrooms_condition == "more_than":
            base_query += " AND numberOfBedRoom >= %s"
        elif bedrooms_condition == "equal":
            base_query += " AND numberOfBedRoom = %s"
        params.append(bedrooms)
    
    if floors is not None and floors_condition is not None:
        if floors_condition == "less_than":
            base_query += " AND numberOfFloor <= %s"
        elif floors_condition == "more_than":
            base_query += " AND numberOfFloor >= %s"
        elif floors_condition == "equal":
            base_query += " AND numberOfFloor = %s"
        params.append(floors)

    if toilets is not None and toilets_condition is not None:
        if toilets_condition == "less_than":
            base_query += " AND numberOfToilet <= %s"
        elif toilets_condition == "more_than":
            base_query += " AND numberOfToilet >= %s"
        elif toilets_condition == "equal":
            base_query += " AND numberOfToilet = %s"
        params.append(toilets)

    if area is not None and area_condition is not None:
        if area_condition == "less_than":
            base_query += " AND landArea <= %s"
        elif area_condition == "more_than":
            base_query += " AND landArea >= %s"
        elif area_condition == "equal":
            base_query += " AND landArea = %s"
        params.append(area)

    if price is not None and price_condition is not None:
        if price_condition == "less_than":
            base_query += " AND price <= %s"
        elif price_condition == "more_than":
            base_query += " AND price >= %s"
        elif price_condition == "equal":
            base_query += " AND price = %s"
        params.append(price)
        
    cursor.execute(base_query, tuple(params))
    results = cursor.fetchall()
    if results:
    # Lấy phần tử chuyển đổi thành số nguyên
      property_ids = [int(result[0]) for result in results]
    else:
        print("No properties found matching your criteria.")
    print(f"result: {property_ids}")
    conn.close()
    print(f"query: {base_query}")
    return property_ids


def extract_numbers(text):
    numbers = re.findall(r'\d+', text)
    return [int(num) for num in numbers]

def extract_conditions(words, index):
    if 'less' in words and 'than' in words and (index > words.index('less') or index < words.index('less')):
        return "less_than"
    elif 'more' in words and 'than' in words and (index > words.index('more') or index < words.index('more')):
        return "more_than"
    elif 'equal' in words and (index > words.index('equal') or index < words.index('equal')):
        return "equal"
    elif 'at' in words and 'least' in words and index > words.index('at') and index > words.index('least'):
        return "more_than"
    elif 'under' in words and (index < words.index('under') or index > words.index('under')):
        return "less_than"
    else:
        return None



def get_bedrooms_floors_toilets_area(text):
    words = text.lower().split()
    numbers = extract_numbers(text)
    print(f"numbers: {numbers}")
    print(f"words: {words}")
    bedrooms, floors, toilets, area, price = None, None, None, None, None
    bedrooms_condition, floors_condition, toilets_condition, area_condition, price_condition = None, None, None, None, None

    if any(word in words for word in ['bedroom', 'bedrooms']):
        bed_index = next(i for i, word in enumerate(words) if word in ['bedroom', 'bedrooms'])
        if bed_index > 0 and numbers:
            bedrooms = numbers.pop(0)
            bedrooms_condition = extract_conditions(words, bed_index)

    if any(word in words for word in ['floor', 'floors']):
        floor_index = next(i for i, word in enumerate(words) if word in ['floor', 'floors'])
        if floor_index > 0 and numbers:
            floors = numbers.pop(0)
            floors_condition = extract_conditions(words, floor_index)

    if any(word in words for word in ['toilet', 'toilets']):
        toilet_index = next(i for i, word in enumerate(words) if word in ['toilet', 'toilets'])
        if toilet_index > 0 and numbers:
            toilets = numbers.pop(0)
            toilets_condition = extract_conditions(words, toilet_index)

    if any(word in words for word in ['area', 'square meters']):
        area_index = next(i for i, word in enumerate(words) if word in ['area', 'square meters'])
        if area_index > 0 and numbers:
            area = numbers.pop(0)
            area_condition = extract_conditions(words, area_index)

    if any(word in words for word in ['price', 'priced', 'prices']):
        price_index = next(i for i, word in enumerate(words) if word in ['price', 'priced', 'prices'])
        if price_index > 0 and numbers:
            price = numbers.pop(0)
            price_condition = extract_conditions(words, price_index)
            
    return bedrooms, bedrooms_condition, floors, floors_condition, toilets, toilets_condition, area, area_condition, price, price_condition


class Message(BaseModel):
    text: str

@app.post("/chatbot")
async def chatbot(message: Message):
    sentence = message.text
    tokenized_sentence = tokenize(sentence)
    X = bag_of_words(tokenized_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "search_property":
                    bedrooms, bedrooms_condition, floors, floors_condition, toilets, toilets_condition, area, area_condition, price, price_condition = get_bedrooms_floors_toilets_area(" ".join(tokenized_sentence))
                    results = query_database(bedrooms, bedrooms_condition, floors, floors_condition, toilets, toilets_condition, area, area_condition, price, price_condition)
                    if results:
                        response = "Here are some properties that match your criteria:\n"
                        for id in results:
                            response += f"https://house-sale-three.vercel.app/details/{id}\n"
                    else:
                        response = "Sorry, I couldn't find any properties that match your criteria."
                    return {"response": response}
                else:
                    return {"response": random.choice(intent['responses'])}
    else:
        return {"response": "I do not understand..."}
