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
        city_info = {
            "name": city_name,
            "lat": data["city"]["cityLatitude"],
            "lng": data["city"]["cityLongitude"],
            "forecastDate": [],
            "maxTemp": [],
            "minTemp": [],
            "weatherIcon": []
        }

        for city_data in data["city"]["forecast"]["forecastDay"]:
            forecast_date = city_data["forecastDate"]
            max_temp = city_data["maxTemp"]
            min_temp = city_data["minTemp"]
            weather_icon = city_data["weatherIcon"]

            city_info["forecastDate"].append(forecast_date)
            city_info["maxTemp"].append(max_temp)
            city_info["minTemp"].append(min_temp)
            city_info["weatherIcon"].append(weather_icon)

        output.append(city_info)



def main():
    output = {"cities": []}  # Initialize the output dictionary with "cities" key

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
                parse_and_append_output(data, city, output["cities"])
                print(f"Processed row {i} of {row_count}")
            else:
                print(f"Skipping row {i} due to previous error.")

    # Write the combined output to a single JSON file without spaces
    with open('world_weather.json', 'w') as output_file:
        json.dump(output, output_file, separators=(',', ':'))

if __name__ == "__main__":
    main()






