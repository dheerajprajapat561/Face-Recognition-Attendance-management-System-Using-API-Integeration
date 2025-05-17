import requests
import json
from datetime import datetime
import random

def post_attendance_swipe(name, employee_id, timestamp, swipe_type):
    url = "https://ddottt6z7ccpe0a-apexdb.adb.me-jeddah-1.oraclecloudapps.com/ords/otrix/oc_hcm_employee_attendance_swips/"
    
    formatted_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").isoformat() + "Z"

    # Generate unique ID using timestamp + randomness
    swipe_id = int(datetime.now().timestamp()) + random.randint(100, 999)

    payload = {
        "id": swipe_id,
        "employee_id": employee_id,
        "swip_time": formatted_timestamp,
        "status": "Open",
        "primary_flag": "Y",
        "company_id": None,
        "app_id": 191,
        "completed_by": None,
        "completed_date": None,
        "cancelled_by": None,
        "cancelled_date": None,
        "row_version": 1,
        "created": formatted_timestamp,
        "created_by": name,
        "updated": formatted_timestamp,
        "updated_by": name,
        "swip_type": swipe_type,
        "latitude": None,
        "longitude": None,
        "employee_name": name
    }

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

    try:
        print("Sending request to API...")
        print(f"Payload: {json.dumps(payload, indent=4)}")

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        print("✅ API Response:", response.status_code, response.json())
        return response

    except requests.exceptions.HTTPError as err:
        print("❌ HTTP Error occurred:")
        print(f"Status Code: {err.response.status_code}")
        print(f"Response Text: {err.response.text}")
    except requests.exceptions.RequestException as e:
        print("❌ API Request failed:")
        print(f"Error: {str(e)}")