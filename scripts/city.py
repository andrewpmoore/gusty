import csv
import requests
import json

def get_weather_data(city_id):
    try:
        url = f"https://worldweather.wmo.int/en/json/{city_id}_en.json"
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for city_id {city_id}: {e}")
        return None

def parse_and_append_output(data, city_name, output):
    if data is not None:  # Only process if data is available
        for city_data in data["city"]["forecast"]["forecastDay"]:
            city_lat = data["city"]["cityLatitude"]
            city_lng = data["city"]["cityLongitude"]
            is_capital = data["city"]["isCapital"]
            src = data["city"]["member"]["url"]
            src_name = data["city"]["member"]["orgName"]

            forecast_date = city_data["forecastDate"]
            max_temp = city_data["maxTemp"]
            min_temp = city_data["minTemp"]
            weather_icon = city_data["weatherIcon"]

            city_info = {
                "name": city_name,
                "lat": city_lat,
                "lng": city_lng,
                "isCapital": is_capital,
                "src": src,
                "srcName": src_name,
                "forecastDay": [
                    {
                        "forecastDate": forecast_date,
                        "maxTemp": max_temp,
                        "minTemp": min_temp,
                        "weatherIcon": weather_icon
                    }
                ]
            }

            if city_name not in output:
                output[city_name] = {"city": []}

            output[city_name]["city"].append(city_info)

def main():
    output = {}  # Initialize the output dictionary
    
    with open('locations.txt', 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # skip header row
        
        row_count = sum(1 for row in reader)  # Count the number of rows
        
        # Reset file pointer to the beginning
        file.seek(0)
        next(reader)  # Skip header row again
        
        for i, row in enumerate(reader, 1):
            country, city, city_id = row
            data = get_weather_data(city_id)
            
            if data is not None:
                parse_and_append_output(data, city, output)
                print(f"Processed row {i} of {row_count}")
            else:
                print(f"Skipping row {i} due to previous error.")

    # Write the combined output to a single JSON file without spaces
    with open('combined_output.json', 'w') as output_file:
        json.dump(output, output_file, separators=(',', ':'))

if __name__ == "__main__":
    main()
