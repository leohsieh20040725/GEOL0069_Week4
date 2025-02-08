# GEOL0069_Satellite Echo Classification
This repository classifies echoes in leads and sea ice using machine learning. It generates an average echo shape, computes the standard deviation for both categories, and compares results against the ESA official classification using a confusion matrix

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

![image](https://github.com/user-attachments/assets/7e91e0b7-e808-4783-9ce5-a81af13b4b38)


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data">Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data</a>
      <ul>
        <li><a href="#Step 0: Read in Functions Needed">Step 0: Read in Functions Needed</a></li>
        <li><a href="#Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)">Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)</a></li>
        <ul>
          <li><a href="#Co-locate the data">Co-locate the data</a></li>
        <li><a href="#Proceeding with Sentinel-3 OLCI Download">Proceeding with Sentinel-3 OLCI Download</a></li>
           <li><a href="#Sentinel-3 SRAL">Sentinel-3 SRAL</a></li>
      </ul>
        </ul>
    </li>
    <li>
     <a href="#Unsupervised Learning">Unsupervised Learning</a>
      <ul>
        <li><a href="#Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]">Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]</a></li>
        <ul>
          <li><a href="#Introduction to K-means Clustering">Introduction to K-means Clustering</a></li>
        <li><a href="#Why K-means for Clustering?">Why K-means for Clustering?</a></li>
        <li><a href="#Key Components of K-means">Key Components of K-means</a></li>
        <li><a href="#The Iterative Process of K-means">The Iterative Process of K-means</a></li>
        <li><a href="#Advantages of K-means">Advantages of K-means</a></li>
        <li><a href="#Basic Code Implementation">Basic Code Implementation</a></li>
          </ul>
        <li><a href="#Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]">Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]</a></li>
        <ul>
        <li><a href="#Introduction to Gaussian Mixture Models">Introduction to Gaussian Mixture Models</a></li>
        <li><a href="#Why Gaussian Mixture Models for Clustering?">Why Gaussian Mixture Models for Clustering?</a></li>
        <li><a href="#Key Components of GMM">Key Components of GMM</a></li>
        <li><a href="#The EM Algorithm in GMM">The EM Algorithm in GMM</a></li>
        <li><a href="#Advantages of GMM">Advantages of GMM</a></li>
        <li><a href="#Basic Code Implementation">Basic Code Implementation</a></li>
           </ul>
        <li><a href="#Image Classification">Image Classification</a></li>
        <ul>
        <li><a href="#K-Means Implementation">K-Means Implementation</a></li>
        <li><a href="#GMM Implementation">GMM Implementation</a></li>
           </ul>
        <li><a href="#Altimetry Classification">Altimetry Classification</a></li>
        <ul>
        <li><a href="#Read in Functions Needed">Read in Functions Needed</a></li>
          </ul>
        <li><a href="#Scatter Plots of Clustered Data">Scatter Plots of Clustered Data</a></li>
        <li><a href="#Waveform Alignment Using Cross-Correlation">Waveform Alignment Using Cross-Correlation</a></li>
        <li><a href="#Compare with ESA data">Compare with ESA data</a></li>   
       
  
  </ol>
</details>





<!-- ABOUT THE PROJECT -->
## About The Project


This project is dedicated to colocating Sentinel-3 (OLCI & SRAL) and Sentinel-2 optical data while leveraging unsupervised learning techniques for classifying sea ice and leads. The goal is to develop an automated pipeline that improves Earth Observation (EO) analysis by fusing different satellite datasets and applying machine learning models to classify environmental features.

Why This Project?
* 🌍 Integrates Sentinel-2 and Sentinel-3 data to improve classification accuracy.
* 🛰️ Leverages machine learning (K-Means & GMM) to identify patterns in EO data.
* 📊 Applies clustering techniques for classifying sea ice and leads.
* 🔄 Automates colocation and processing, reducing manual effort.
* ✅ Compares model predictions with ESA’s official classification for validation.

Key Learning Outcomes
* Understanding the colocation process for multi-sensor satellite data.
* Applying unsupervised learning to classify environmental features.
* Using altimetry and optical data fusion for sea ice and lead classification.
* Evaluating classification performance through confusion matrices and accuracy metrics.

Of course, satellite data analysis is an evolving field, and different approaches may work for different datasets. This project serves as a practical guide to combining remote sensing and machine learning. Contributions and improvements are always welcome! 🚀

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

This project utilizes key Python libraries and geospatial tools to process, analyze, and classify Earth Observation (EO) data. Below are the major dependencies used:

* NumPy – Numerical computations and matrix operations
* Pandas – Data manipulation and tabular processing
* Matplotlib – Visualization of classification results
* Rasterio – Handling Sentinel-2 geospatial raster data
* netCDF4 – Processing Sentinel-3 altimetry data
* Scikit-Learn – Machine learning models (K-Means, GMM)
* Folium – Geospatial data visualization
* Shapely – Geometric operations for colocation analysis
* Requests – API calls for Sentinel-3 metadata retrieval

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data -->
## Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data
This section delves into the colocation of Sentinel-3 and Sentinel-2 data, leveraging their complementary strengths to enhance Earth Observation analysis. By combining the high spatial resolution of Sentinel-2 with the extensive coverage and altimetry insights of Sentinel-3, we can achieve a more comprehensive and detailed understanding of Earth's surface features.

This fusion of datasets enables precise environmental monitoring, making it particularly valuable for applications such as sea ice and lead classification.

In the following sections, we will walk through the key steps to identify, align, and analyze these datasets effectively.

### Step 0: Read in Functions Needed

To efficiently fetch and process our data, we will begin by loading the essential functions. These functions are the same as those used in the data_fetching notebook from Week 3. Their primary purpose is to retrieve metadata for the two satellites of interest, ensuring a seamless and structured data acquisition process.


```python
from google.colab import drive
drive.mount('/content/drive')
```
This code mounts Google Drive to the Google Colab environment, allowing direct access to files stored in Drive. The from google.colab import drive statement imports the necessary module, and drive.mount('/content/drive') links the Drive to the Colab session at /content/drive. When executed, it prompts the user for authorization, requiring them to enter an authentication code. Once mounted, users can seamlessly read, write, and manage files from their Google Drive as if they were local files within the Colab notebook.



```python
from datetime import datetime, timedelta
from shapely.geometry import Polygon, Point, shape
import numpy as np
import requests
import pandas as pd
from xml.etree import ElementTree as ET
import os
import json
import folium


def make_api_request(url, method="GET", data=None, headers=None):
    global access_token
    if not headers:
        headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.request(method, url, json=data, headers=headers)
    if response.status_code in [401, 403]:
        global refresh_token
        access_token = refresh_access_token(refresh_token)
        headers["Authorization"] = f"Bearer {access_token}"
        response = requests.request(method, url, json=data, headers=headers)
    return response


def query_sentinel3_olci_arctic_data(start_date, end_date, token):
    """
    Queries Sentinel-3 OLCI data within a specified time range from the Copernicus Data Space,
    targeting data collected over the Arctic region.

    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    token (str): Access token for authentication.

    Returns:
    DataFrame: Contains details about the Sentinel-3 OLCI images.
    """

    all_data = []
    arctic_polygon = "POLYGON((-180 60, 180 60, 180 90, -180 90, -180 60))"
    # arctic_polygon = (
    #     "POLYGON ((-81.7 71.7, -81.7 73.8, -75.1 73.8, -75.1 71.7, -81.7 71.7))"
    # )

    filter_string = (
        f"Collection/Name eq 'SENTINEL-3' and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'OL_1_EFR___') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T23:59:59.999Z"
    )

    next_url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter={filter_string} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{arctic_polygon}')&"
        f"$top=1000"
    )

    headers = {"Authorization": f"Bearer {token}"}

    while next_url:
        response = make_api_request(next_url, headers=headers)
        if response.status_code == 200:
            data = response.json()["value"]
            all_data.extend(data)
            next_url = response.json().get("@odata.nextLink")
        else:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break

    return pd.DataFrame(all_data)


def get_access_and_refresh_token(username, password):
    """Retrieve both access and refresh tokens."""
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    tokens = response.json()
    return tokens["access_token"], tokens["refresh_token"]


def refresh_access_token(refresh_token):
    """Attempt to refresh the access token using the refresh token."""
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": "cdse-public",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()  # This will throw an error for non-2xx responses
        return response.json()["access_token"]
    except requests.exceptions.HTTPError as e:
        print(f"Failed to refresh token: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 400:
            print("Refresh token invalid, attempting re-authentication...")
            # Attempt to re-authenticate
            username = username
            password = password
            # This requires securely managing the credentials, which might not be feasible in all contexts
            access_token, new_refresh_token = get_access_and_refresh_token(
                username, password
            )  # This is a placeholder
            refresh_token = (
                new_refresh_token  # Update the global refresh token with the new one
            )
            return access_token
        else:
            raise

def download_single_product(
    product_id, file_name, access_token, download_dir="downloaded_products"
):
    """
    Download a single product from the Copernicus Data Space.

    :param product_id: The unique identifier for the product.
    :param file_name: The name of the file to be downloaded.
    :param access_token: The access token for authorization.
    :param download_dir: The directory where the product will be saved.
    """
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Construct the download URL
    url = (
        f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    )

    # Set up the session and headers
    headers = {"Authorization": f"Bearer {access_token}"}
    session = requests.Session()
    session.headers.update(headers)

    # Perform the request
    response = session.get(url, headers=headers, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Define the path for the output file
        output_file_path = os.path.join(download_dir, file_name + ".zip")

        # Stream the content to a file
        with open(output_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded: {output_file_path}")
    else:
        print(
            f"Failed to download product {product_id}. Status Code: {response.status_code}"
        )

def query_sentinel3_sral_arctic_data(start_date, end_date, token):
    """
    Queries Sentinel-3 SRAL data within a specified time range from the Copernicus Data Space,
    targeting data collected over the Arctic region.

    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    token (str): Access token for authentication.

    Returns:
    DataFrame: Contains details about the Sentinel-3 SRAL images.
    """

    all_data = []
    # arctic_polygon = "POLYGON((-180 60, 180 60, 180 90, -180 90, -180 60))"
    arctic_polygon = (
        "POLYGON ((-81.7 71.7, -81.7 73.8, -75.1 73.8, -75.1 71.7, -81.7 71.7))"
    )

    filter_string = (
        f"Collection/Name eq 'SENTINEL-3' and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'SR_2_LAN_SI') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T23:59:59.999Z"
    )

    next_url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter={filter_string} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{arctic_polygon}')&"
        f"$top=1000"
    )

    headers = {"Authorization": f"Bearer {token}"}

    while next_url:
        response = make_api_request(
            next_url, headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            data = response.json()["value"]
            all_data.extend(data)
            next_url = response.json().get("@odata.nextLink")
        else:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break

    return pd.DataFrame(all_data)


def query_sentinel2_arctic_data(
    start_date,
    end_date,
    token,
    min_cloud_percentage=10,
    max_cloud_percentage=50,
):
    """
    Queries Sentinel-2 data within a specified time range from the Copernicus Data Space,
    considering a range of cloud coverage by treating greater than and less than conditions as separate attributes.
    Handles pagination to fetch all available data.

    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    token (str): Access token for authentication.
    min_cloud_percentage (int): Minimum allowed cloud coverage.
    max_cloud_percentage (int): Maximum allowed cloud coverage.

    Returns:
    DataFrame: Contains details about the Sentinel-2 images.
    """

    all_data = []
    arctic_polygon = "POLYGON((-180 60, 180 60, 180 90, -180 90, -180 60))"

    filter_string = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value ge {min_cloud_percentage}) and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value le {max_cloud_percentage}) and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T23:59:59.999Z"
    )

    next_url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter={filter_string} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{arctic_polygon}')&"
        f"$top=1000"
    )

    headers = {"Authorization": f"Bearer {token}"}

    while next_url:
        response = make_api_request(
            next_url, headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            data = response.json()["value"]
            all_data.extend(data)
            next_url = response.json().get("@odata.nextLink")
        else:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break

    return pd.DataFrame(all_data)


def plot_results(results):
    m = folium.Map(location=[0, 0], zoom_start=2)
    for idx, row in results.iterrows():
        try:
            geojson1 = json.loads(row["Satellite1_Footprint"].replace("'", '"'))
            geojson2 = json.loads(row["Satellite2_Footprint"].replace("'", '"'))

            folium.GeoJson(geojson1, name=row["Satellite1_Name"]).add_to(m)
            folium.GeoJson(geojson2, name=row["Satellite2_Name"]).add_to(m)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    folium.LayerControl().add_to(m)
    return m


def parse_geofootprint(footprint):
    """
    Parses a JSON-like string to extract the GeoJSON and convert to a Shapely geometry.
    """
    try:
        geo_json = json.loads(footprint.replace("'", '"'))
        return shape(geo_json)
    except json.JSONDecodeError:
        return None


def check_collocation(
    df1, df2, start_date, end_date, time_window=pd.to_timedelta("1 day")
):

    collocated = []
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    for idx1, row1 in df1.iterrows():
        footprint1 = parse_geofootprint(row1["GeoFootprint"])
        if footprint1 is None:
            continue

        s1_start = row1["ContentDate.Start"]
        s1_end = row1["ContentDate.End"]

        if s1_end < start_date or s1_start > end_date:
            continue

        s1_start_adjusted = s1_start - time_window
        s1_end_adjusted = s1_end + time_window

        for idx2, row2 in df2.iterrows():
            footprint2 = parse_geofootprint(row2["GeoFootprint"])
            if footprint2 is None:
                continue

            s2_start = row2["ContentDate.Start"]
            s2_end = row2["ContentDate.End"]

            if s2_end < start_date or s2_start > end_date:
                continue
            if max(s1_start_adjusted, s2_start) <= min(s1_end_adjusted, s2_end):
                if footprint1.intersects(footprint2):
                    collocated.append(
                        {
                            "Satellite1_Name": row1["Name"],
                            "Satellite1_ID": row1["Id"],
                            "Satellite1_Footprint": row1["GeoFootprint"],
                            "Satellite2_Name": row2["Name"],
                            "Satellite2_ID": row2["Id"],
                            "Satellite2_Footprint": row2["GeoFootprint"],
                            "Overlap_Start": max(
                                s1_start_adjusted, s2_start
                            ).isoformat(),
                            "Overlap_End": min(s1_end_adjusted, s2_end).isoformat(),
                        }
                    )

    return pd.DataFrame(collocated)


def make_timezone_naive(dt):
    """Convert a timezone-aware datetime object to timezone-naive in local time."""
    return dt.replace(tzinfo=None)
```
This script is designed to fetch, process, and analyze Sentinel-2 and Sentinel-3 Earth observation data from the Copernicus Data Space Ecosystem. It first imports essential libraries like requests, pandas, shapely, and folium to handle API requests, process geospatial data, and visualize satellite footprints. The script manages authentication by obtaining an access token and refreshing it when needed. It includes functions to query Sentinel-3 OLCI (ocean and land color) and SRAL (radar altimeter) data, as well as Sentinel-2 optical data, filtering results by date range, geographic location, and cloud cover percentage. Additionally, it provides a downloading feature for Sentinel products based on unique product IDs. The script processes geospatial footprints, identifying and matching images based on geographic overlap, and visualizes results using interactive maps. Time handling functions ensure correct formatting of timestamps for accurate data retrieval. Overall, this script serves as a complete data pipeline for retrieving, processing, and analyzing Sentinel satellite data, enabling seamless integration with scientific research and Earth observation projects. 🚀

<!-- Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case) -->
### Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)
In this step, we demonstrate the process of co-locating Sentinel-2 and Sentinel-3 OLCI data by first retrieving their metadata, following the same approach used in Week 3. Since our objective is to identify common locations observed by both satellites, we extract metadata for each separately. This results in two distinct datasets representing the respective satellites, stored as sentinel3_olci_data and sentinel2_data. These metadata tables will serve as the foundation for aligning and analyzing the satellite observations effectively.



```python
username = ""
password = ""
access_token, refresh_token = get_access_and_refresh_token(username, password)
start_date = "2018-06-01"
end_date = "2018-06-02"
path_to_save_data = "/content/drive/MyDrive/GEOL0069/2425/Week 4/" # Here you can edit where you want to save your metadata
s3_olci_metadata = query_sentinel3_olci_arctic_data(
    start_date, end_date, access_token
)

s2_metadata = query_sentinel2_arctic_data(
    start_date,
    end_date,
    access_token,
    min_cloud_percentage=0,
    max_cloud_percentage=10,
)

# You can also save the metadata
s3_olci_metadata.to_csv(
    path_to_save_data+"sentinel3_olci_metadata.csv",
    index=False,
)

s2_metadata.to_csv(
    path_to_save_data+"sentinel2_metadata.csv",
    index=False,
)
```
This code retrieves and saves metadata for Sentinel-2 and Sentinel-3 OLCI satellite data within a specified date range. It first initializes user credentials (username and password), which are required to obtain an access token and refresh token using the get_access_and_refresh_token() function. The user then defines a start date and end date for the data query, as well as a file path where the retrieved metadata will be saved.

Next, the script calls query_sentinel3_olci_arctic_data() to fetch Sentinel-3 OLCI data and query_sentinel2_arctic_data() to fetch Sentinel-2 optical data for the given time period. For Sentinel-2, an additional filter is applied to select images with cloud coverage between 0% and 10%, ensuring better visibility of the Earth's surface.

Finally, the retrieved metadata is saved as CSV files in the specified directory. The Sentinel-3 OLCI metadata is stored as sentinel3_olci_metadata.csv, while the Sentinel-2 metadata is saved as sentinel2_metadata.csv. These files serve as a structured dataset that can be used for further analysis, such as co-locating observations from both satellites.

```python
from IPython.display import display

display(s3_olci_metadata)
```
This code is used to display the Sentinel-3 OLCI metadata in a structured table format within a Jupyter Notebook or Google Colab environment. It first imports the display function from the IPython library, which allows for better visualization of data compared to the standard print() function. The display(s3_olci_metadata) command then renders the previously retrieved Sentinel-3 OLCI metadata as a well-formatted table, making it easier to analyze and interpret. This approach is particularly useful when working with large datasets, as it ensures the metadata is displayed in a clear and organized manner rather than raw text output.
![422d13d80a9193db1c3d56e377ac803](https://github.com/user-attachments/assets/cfcaa103-e028-45e3-bcf9-b91b6957116c)
The table displays the metadata retrieved for Sentinel-3 OLCI images within the specified time range. It includes essential attributes such as unique product IDs, names, content types, origin dates, modification dates, and storage paths. This metadata is crucial for identifying and accessing relevant satellite data for further analysis and co-location with Sentinel-2.

```python
from IPython.display import display

display(s2_metadata)
```
This code utilizes the display function from the IPython.display module to visually present the s2_metadata DataFrame. The s2_metadata DataFrame contains metadata for Sentinel-2 images retrieved from the Copernicus Data Space within the specified date range. This metadata includes key details such as product IDs, acquisition times, geospatial footprints, and cloud cover percentages. By using display(s2_metadata), the dataset is rendered in a structured tabular format, making it easier to inspect, analyze, and verify the retrieved information directly within an interactive notebook environment.
![7a75d1a91255c7e1a9145b45bb2fb72](https://github.com/user-attachments/assets/b428d280-5b6a-4ce4-a588-f9d46936ab1d)
This table displays metadata retrieved for Sentinel-2 images using the Copernicus Data Space API. It includes details such as product IDs, content type, content length, acquisition dates, publication and modification timestamps, online availability, and storage paths. This dataset is essential for analyzing and identifying relevant Sentinel-2 imagery based on specific timeframes and geospatial locations.

#### Co-locate the data
In this section we use the metadata we have just produced to produce the co-location pair details. The logic of the code is match rows from S2 and S3 OLCI by their geo_footprint.
```python
s3_olci_metadata = pd.read_csv(
    path_to_save_data + "sentinel3_olci_metadata.csv"
)
s2_metadata = pd.read_csv(
    path_to_save_data + "sentinel2_metadata.csv"
)
```
This code snippet loads previously saved metadata for Sentinel-3 OLCI and Sentinel-2 imagery into Pandas DataFrames. The pd.read_csv() function reads the metadata from CSV files stored in the specified directory, allowing for easy access and further analysis. The s3_olci_metadata variable contains details about Sentinel-3 OLCI data, while s2_metadata holds information on Sentinel-2 data. By loading these datasets, we can efficiently process and analyze satellite imagery without needing to re-fetch the data from the API, saving both time and computational resources.
```python
s3_olci_metadata["ContentDate.Start"] = pd.to_datetime(
    s3_olci_metadata["ContentDate"].apply(lambda x: eval(x)["Start"])
).apply(make_timezone_naive)
s3_olci_metadata["ContentDate.End"] = pd.to_datetime(
    s3_olci_metadata["ContentDate"].apply(lambda x: eval(x)["End"])
).apply(make_timezone_naive)

s2_metadata["ContentDate.Start"] = pd.to_datetime(
    s2_metadata["ContentDate"].apply(lambda x: eval(x)["Start"])
).apply(make_timezone_naive)
s2_metadata["ContentDate.End"] = pd.to_datetime(
    s2_metadata["ContentDate"].apply(lambda x: eval(x)["End"])
).apply(make_timezone_naive)

results = check_collocation(
    s2_metadata, s3_olci_metadata, start_date, end_date,time_window=pd.to_timedelta("10 minutes")
)
```
This code processes the metadata for Sentinel-3 OLCI and Sentinel-2 imagery by extracting and converting the content dates into a standardized datetime format. The ContentDate column, which contains structured date information, is evaluated using eval() to extract the start and end times. The pd.to_datetime() function ensures that these extracted timestamps are properly formatted, and the make_timezone_naive() function is applied to remove any timezone awareness, ensuring consistency in time comparisons.

Once the date conversions are complete, the check_collocation() function is used to identify instances where the Sentinel-2 and Sentinel-3 OLCI data overlap within a defined time window of 10 minutes. This step is crucial for analyzing co-located satellite observations, ensuring that datasets from both satellites are temporally aligned for further geospatial analysis. The resulting results DataFrame contains matched records where both Sentinel-2 and Sentinel-3 observations occurred within the specified timeframe, allowing for effective data fusion and comparison.
```python
from IPython.display import display

display(results.head(5))
```
![image](https://github.com/user-attachments/assets/f136ac9d-181f-42e3-9079-af0e87e2fdbf)
The table displays the first five rows of the collocated dataset, showing matched Sentinel-2 and Sentinel-3 OLCI observations. Each row contains details about the two satellites, including their unique IDs, footprints (geographical coverage), and the time range during which their observations overlap within a 10-minute window. This output helps verify the successful identification of collocated satellite data for further analysis.

```python
from IPython.display import display

map_result = plot_results(results.head(5))
display(map_result)
```


This code generates and displays an interactive map visualization of the first five collocated satellite observations. The plot_results function takes the top five rows from the results DataFrame, which contains information about overlapping Sentinel-2 and Sentinel-3 OLCI observations. It then plots the footprints of both satellites on a map using the folium library. The display function from IPython.display ensures that the generated map is rendered within the notebook interface, allowing users to visually inspect the geographic locations where the satellite observations overlap.

![image](https://github.com/user-attachments/assets/e0bb2933-f303-4a27-ace3-e0dab44e97eb)
This interactive map visualization displays the geographical footprints of the first five collocated satellite observations from Sentinel-2 and Sentinel-3 OLCI. The overlapping satellite data areas are highlighted, showing the regions where both satellites have captured observations within the specified time window.


<!-- Proceeding with Sentinel-3 OLCI Download -->
#### Proceeding with Sentinel-3 OLCI Download
Next, we shift our focus to retrieving Sentinel-3 OLCI data. This process follows the same structured approach used for Sentinel-2, ensuring consistency in methodology. By applying the same filename conversion logic, we systematically access and download the required datasets from the Copernicus Dataspace. This step ensures seamless integration of Sentinel-3 OLCI data into our analysis pipeline.

```python
download_dir = ""  # Replace with your desired download directory
product_id = results['Satellite1_ID'][0] # Replace with your desired file id
file_name = results['Satellite1_Name'][0]# Replace with your desired filename
# Download the single product
download_single_product(product_id, file_name, access_token, download_dir)
```
This code snippet facilitates the download of a specific Sentinel-3 OLCI product from the Copernicus Dataspace. The download_dir variable is used to specify the target directory where the downloaded file will be saved. The product_id and file_name are extracted from the results dataframe, selecting the first product entry for download. The download_single_product function is then called with these parameters, along with an access_token for authentication. This ensures secure and structured retrieval of the satellite data, storing it in the designated directory for further analysis. Users can modify the product_id, file_name, and download_dir to customize their download preferences.

#### Sentinel-3 SRAL
In addition to co-locating Sentinel-2 and Sentinel-3 OLCI data, we can also integrate Sentinel-3 SRAL altimetry data. The overall approach remains the same, requiring only the retrieval of the S3 SRAL metadata. By incorporating SRAL data, we enhance our dataset with valuable altimetry measurements, enabling a more comprehensive analysis of surface characteristics.

```python
sentinel3_sral_data = query_sentinel3_sral_arctic_data(
    start_date, end_date, access_token
)

sentinel3_sral_data.to_csv(
    path_to_save_data + "s3_sral_metadata.csv",
    index=False,
)
```
This code retrieves Sentinel-3 SRAL (Synthetic Aperture Radar Altimeter) metadata for a specified time range and saves it as a CSV file. The query_sentinel3_sral_arctic_data function is used to fetch the SRAL data from the Copernicus Data Space, using the defined start_date, end_date, and access_token for authentication. Once the metadata is retrieved, it is stored in the variable sentinel3_sral_data. The second part of the code converts this dataset into a CSV file named "s3_sral_metadata.csv", which is saved in the specified directory defined by path_to_save_data. This ensures that the retrieved SRAL metadata is easily accessible for further processing or analysis.

And now you do the co-locaton again for S3 SRAL with S2 for example:
```python
s3_sral_metadata = pd.read_csv(
    path_to_save_data + "s3_sral_metadata.csv"
)
s2_metadata = pd.read_csv(
    path_to_save_data + "sentinel2_metadata.csv"
)
```
This code loads previously saved metadata files for Sentinel-3 SRAL and Sentinel-2 satellite data into Pandas DataFrames for further analysis. The pd.read_csv function reads the "s3_sral_metadata.csv" file, which contains metadata for Sentinel-3 SRAL data, and stores it in the variable s3_sral_metadata. Similarly, the "sentinel2_metadata.csv" file, which holds metadata for Sentinel-2 data, is read and stored in the variable s2_metadata. These datasets will be used in subsequent processing steps, such as identifying co-locations between the two datasets for comparative or combined analysis.


```python
s3_sral_metadata["ContentDate.Start"] = pd.to_datetime(
    s3_sral_metadata["ContentDate"].apply(lambda x: eval(x)["Start"])
).apply(make_timezone_naive)
s3_sral_metadata["ContentDate.End"] = pd.to_datetime(
    s3_sral_metadata["ContentDate"].apply(lambda x: eval(x)["End"])
).apply(make_timezone_naive)

s2_metadata["ContentDate.Start"] = pd.to_datetime(
    s2_metadata["ContentDate"].apply(lambda x: eval(x)["Start"])
).apply(make_timezone_naive)
s2_metadata["ContentDate.End"] = pd.to_datetime(
    s2_metadata["ContentDate"].apply(lambda x: eval(x)["End"])
).apply(make_timezone_naive)

results = check_collocation(
    s2_metadata, s3_sral_metadata, start_date, end_date,time_window=pd.to_timedelta("10 minutes")
)
```
This code processes the metadata for Sentinel-3 SRAL and Sentinel-2 datasets by extracting and converting their content date fields into a standardized datetime format for accurate time-based analysis. It begins by applying the eval function to parse the "ContentDate" column in both datasets, extracting the "Start" and "End" timestamps. The extracted values are then converted into timezone-naive datetime objects using pd.to_datetime, followed by the make_timezone_naive function to ensure consistency in time representation.

Once the timestamps are properly formatted, the code performs a collocation check between Sentinel-2 and Sentinel-3 SRAL datasets. The check_collocation function is used to determine overlapping observations based on spatial and temporal criteria. The time_window=pd.to_timedelta("10 minutes") argument ensures that only those data points that fall within a 10-minute time difference are considered collocated. The output is stored in the results variable, which contains details of matching observations between Sentinel-2 and Sentinel-3 SRAL data.


And now you can plot the co-location results again:
```python
from IPython.display import display

map_result = plot_results(results.head(5))
display(map_result)
```
This code is responsible for visualizing the collocation results between Sentinel-2 and Sentinel-3 SRAL datasets on an interactive map. It first extracts the top five collocated entries from the results dataframe using results.head(5), ensuring that only a subset of the data is plotted for clarity.

The plot_results function is then called with this subset, generating a map visualization that highlights the geographical footprints of the collocated satellite observations. This function likely uses GeoJSON data to overlay footprints onto a world map, making it easier to interpret spatial relationships between observations.

Finally, the generated map is displayed using the display function from IPython.display, which ensures that the interactive map renders correctly within environments such as Jupyter Notebook or Google Colab. This visualization helps in assessing the accuracy and distribution of the collocated satellite data.



![image](https://github.com/user-attachments/assets/908fe20f-02df-403e-9937-32f8b527bc1b)
This interactive map visualizes the collocation of Sentinel-2 and Sentinel-3 SRAL satellite data. The blue outlines represent the geographical footprints of the detected overlaps, illustrating how the two satellite datasets align over the Arctic region. This visualization helps assess spatial intersections and validate the effectiveness of the collocation process.


<!-- Unsupervised Learning -->
## Unsupervised Learning
This section introduces a crucial domain in machine learning and AI: unsupervised learning. Rather than diving deep into theoretical complexities, our focus is on providing a hands-on, practical guide. Our goal is to equip you with the knowledge and tools necessary to effectively apply unsupervised learning techniques to real-world Earth Observation (EO) scenarios.

While unsupervised learning has a vast range of applications, this discussion will primarily focus on classification tasks. These techniques are particularly powerful in detecting patterns and grouping data when predefined labels are unavailable. By leveraging these methods, you’ll develop an intuitive understanding of how to uncover hidden structures and relationships within your datasets, even in the absence of explicit categorizations.

In this notebook, we will tackle two key tasks:

1-Distinguishing sea ice from leads using image classification based on Sentinel-2 optical data.

2-Identifying sea ice and leads through altimetry data classification utilizing Sentinel-3 altimetry data.

By the end of this section, you will have a solid foundation in applying unsupervised learning for remote sensing and EO data analysis.

<!-- Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006] -->
### Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]
#### Introduction to K-means Clustering
K-means clustering is a widely used unsupervised learning algorithm designed to partition a dataset into k distinct groups (clusters). The parameter k, which represents the number of clusters, is predefined by the user. The algorithm classifies data points based on feature similarity, ensuring that data points within the same cluster are more alike than those in different clusters [MacQueen et al., 1967].

At its core, K-means clustering works by:

1-Defining k centroids, one for each cluster.

2-Assigning each data point to the nearest centroid based on similarity.

3-Updating the centroids by recalculating the cluster means.

4-Iterating this process until the centroids remain stable, minimizing the variation within clusters.

This method is particularly useful for pattern recognition, data segmentation, and exploratory data analysis, making it a fundamental tool in unsupervised learning.
#### Why K-means for Clustering?
K-means clustering is particularly effective in scenarios where:

1-Unknown Data Structure: The algorithm does not require prior knowledge of the data distribution or structure, making it an excellent choice for exploratory data analysis and pattern detection.

2-Efficiency and Scalability: K-means is computationally efficient, easy to implement, and capable of handling large datasets with minimal complexity, making it a preferred choice for various real-world applications.

#### Key Components of K-means
1-Choosing K: The number of clusters (k) must be predefined before running the algorithm, which can significantly impact the clustering results.

2-Centroid Initialization: The initial placement of cluster centroids plays a crucial role in determining the final clustering outcome.

3-Assignment Step: Each data point is assigned to the closest centroid based on the squared Euclidean distance, effectively grouping similar points together.

4-Update Step: The centroids are recalculated as the mean position of all data points assigned to their respective clusters.

#### The Iterative Process of K-means
K-means follows an iterative approach where the assignment and update steps repeat until the centroids stabilize, meaning they no longer change significantly. This process minimizes intra-cluster variation and ensures convergence to an optimal clustering solution, though it may sometimes settle on a local optimum.

#### Advantages of K-means
1-High Efficiency: K-means is computationally fast, making it well-suited for handling large datasets.

2-Ease of Interpretation: The clustering results are straightforward, making it easy to analyze and understand the underlying data patterns.

#### Basic Code Implementation
Below is a fundamental implementation of the K-means clustering algorithm. This example provides a solid starting point for understanding how the algorithm works and can be adapted for various data analysis tasks. By exploring this implementation, you can gain insights into clustering techniques and apply them effectively to real-world datasets.

```python
from google.colab import drive
drive.mount('/content/drive')
```

This code is used in Google Colab to mount Google Drive, allowing users to access files stored in their Drive directly from the Colab notebook. The from google.colab import drive statement imports the drive module from the google.colab package, which is specifically designed for working with Google Colab. The drive.mount('/content/drive') command then mounts the user’s Google Drive to the /content/drive directory, enabling seamless reading and writing of files. Once executed, users will be prompted to authenticate their Google account and grant necessary permissions. This is particularly useful for loading datasets, saving outputs, and managing files within Colab.

```python
pip install rasterio
```
The command pip install rasterio is used to install the Rasterio library, a powerful tool for reading, writing, and processing geospatial raster data. Rasterio provides a Pythonic interface to interact with raster datasets, such as satellite imagery and digital elevation models, using the GDAL (Geospatial Data Abstraction Library) backend. By running this command, users can install Rasterio and its dependencies, enabling them to work with raster data efficiently, perform geospatial analysis, and manipulate multi-band imagery in Python. This command is typically executed in a Jupyter Notebook, Google Colab, or a terminal to ensure that the library is available for use in geospatial projects.

```python
pip install netCDF4
```
The command pip install netCDF4 is used to install the netCDF4 library, which provides tools for reading, writing, and manipulating netCDF (Network Common Data Form) files. NetCDF is a widely used data format for storing large-scale scientific data, particularly in meteorology, oceanography, and climate research. This library enables users to efficiently handle multi-dimensional data arrays, access metadata, and perform various operations on netCDF files in Python. Running this command ensures that the necessary dependencies are installed, allowing seamless interaction with netCDF datasets in Jupyter Notebooks, Google Colab, or Python scripts.


```python
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
```
This Python script implements a basic K-means clustering algorithm using the scikit-learn library. It begins by importing necessary libraries, including KMeans for clustering, matplotlib.pyplot for visualization, and numpy for numerical operations. The script generates a sample dataset consisting of 100 random points in a two-dimensional space. A K-means model is then initialized with four clusters (n_clusters=4) and trained on this dataset using kmeans.fit(X). After training, the model predicts the cluster assignments for each data point, storing the results in y_kmeans. To visualize the clustering results, a scatter plot is created where each point is color-coded according to its assigned cluster. Additionally, the computed centroids of the clusters are extracted using kmeans.cluster_centers_ and displayed as black dots to indicate the center of each group. Finally, the plt.show() function renders the plot, providing a clear visualization of how the algorithm has grouped the data into distinct clusters. This implementation serves as a foundational example for understanding how K-means clustering can be applied to pattern recognition and data segmentation tasks.

![image](https://github.com/user-attachments/assets/e336776c-92d6-4d6a-b3fc-be0c1f41960d)

Visualization of K-means clustering results on a randomly generated dataset. The colored points represent individual data samples grouped into four clusters, while the black dots indicate the centroids of each cluster, calculated by the K-means algorithm.

<!-- Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006] -->
### Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]
#### Introduction to Gaussian Mixture Models
Gaussian Mixture Models (GMM) are a powerful probabilistic approach for modeling normally distributed subpopulations within a larger dataset. This technique assumes that data is generated from a combination of multiple Gaussian distributions, each characterized by its unique mean and variance [Reynolds and others, 2009]. GMMs are widely employed in clustering and density estimation, offering a structured way to represent intricate data distributions by merging simpler components.

#### Why Gaussian Mixture Models for Clustering?
Gaussian Mixture Models are particularly advantageous in clustering applications where:

1-Soft Clustering Capabilities: Unlike K-means, which assigns each data point to a single cluster, GMM provides a probabilistic classification. This means each data point has a probability of belonging to multiple clusters, enabling a more nuanced and flexible clustering approach while accounting for uncertainty.

2-Adaptive Cluster Shapes: Unlike K-means, which assumes clusters are spherical, GMM accommodates clusters of varying shapes and sizes. By adjusting the covariance structure of each Gaussian component, GMM provides a more adaptable and precise clustering solution for complex datasets.

These features make GMM a robust choice for clustering scenarios where data exhibits overlapping distributions or varying density regions.

#### Key Components of GMM
1-Defining the Number of Components (Gaussians):
Similar to specifying the number of clusters in K-means, GMM requires determining the number of Gaussian components. This defines how many distinct distributions will be used to model the data.

2-Expectation-Maximization (EM) Algorithm:
GMM utilizes the EM algorithm to iteratively refine the model. It alternates between estimating the probability of each data point belonging to a Gaussian and updating the parameters (mean, variance, and weight) to maximize the model’s likelihood.

3-Covariance Structure:
The shape and orientation of clusters are influenced by the covariance type of the Gaussians. This allows for flexibility, supporting spherical, diagonal, tied, or fully adaptable cluster shapes.



#### The EM Algorithm in GMM
The EM algorithm follows an iterative two-step process to optimize clustering:

1-Expectation Step (E-step):
Assigns probabilities to each data point, estimating the likelihood that it belongs to a particular Gaussian component.

2-Maximization Step (M-step):
Updates the Gaussian parameters (mean, variance, and weight) to maximize the overall likelihood of the dataset given the current assignments.

This process repeats until convergence, meaning the parameters stabilize and the model achieves an optimal fit.


#### Advantages of GMM
1-Probabilistic Soft Clustering:
Unlike hard clustering methods like K-means, GMM provides a probability score for each data point’s cluster membership. This helps in capturing uncertainty and overlapping group structures.

2-Flexible Cluster Shapes:
GMM supports non-spherical cluster formations, making it well-suited for datasets where clusters have varying sizes, orientations, or densities.

By leveraging GMM’s adaptability and probabilistic framework, it becomes an excellent choice for clustering complex datasets with overlapping distributions.

#### Basic Code Implementation
Below is a fundamental implementation of the Gaussian Mixture Model (GMM). This example provides a foundational understanding of how GMM works and serves as a practical starting point for incorporating it into data analysis tasks.


```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()
```
This code demonstrates the implementation of the Gaussian Mixture Model (GMM) for clustering a set of randomly generated data points. It begins by importing necessary libraries: GaussianMixture from sklearn.mixture for clustering, matplotlib.pyplot for visualization, and numpy for numerical operations. The dataset consists of 100 randomly generated points in a two-dimensional space. A GMM model is then initialized with three components (clusters) and fitted to the dataset. The model predicts cluster assignments for each data point. To visualize the results, the data points are plotted using different colors to represent different clusters, and the computed cluster centers (means) are highlighted in black. This visualization helps illustrate how GMM effectively groups similar data points based on probabilistic distributions.

![image](https://github.com/user-attachments/assets/ba6c9c55-ed2b-43ef-82a8-543445e64478)
Visualization of clustering results using the Gaussian Mixture Model (GMM). The data points are grouped into three distinct clusters, each represented by a different color. The black points indicate the computed cluster centers (means), highlighting the probabilistic nature of GMM clustering.

### Image Classification
In this section, we delve into the application of unsupervised learning techniques for image classification. Our primary focus is distinguishing between sea ice and leads using Sentinel-2 imagery. By leveraging clustering algorithms, we can identify and classify patterns in the imagery without the need for labeled data, enhancing our ability to analyze and interpret remote sensing data efficiently.

#### K-Means Implementation
```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()

del kmeans, labels, band_data, band_stack, valid_data_mask, X, labels_image
```
This Python script performs K-means clustering on a Sentinel-2 satellite image to classify surface features based on reflectance values from the red (B4) band. It begins by importing necessary libraries: rasterio for handling satellite imagery, numpy for numerical operations, sklearn.cluster.KMeans for clustering, and matplotlib.pyplot for visualization. The script specifies the file paths for the Sentinel-2 image stored in a designated directory and loads Band 4 (Red) using rasterio.open(), storing the pixel values in a NumPy array. To ensure meaningful processing, a valid data mask is created to filter out pixels with no-data values (zero pixels).

Once the image data is loaded, the script reshapes the band values into a 1D array while excluding invalid pixels. The K-means clustering algorithm is then applied with two clusters (n_clusters=2), aiming to distinguish different surface types, such as sea ice and open water. After fitting the model, the script assigns each pixel a cluster label, classifying the satellite image into distinct regions based on reflectance characteristics.

To reconstruct and visualize the results, the classified pixel labels are reshaped back into the original image dimensions. Any masked-out pixels are assigned a no-data value (-1) to preserve data integrity. The clustering results are displayed using plt.imshow(), where different clusters are represented in distinct colors, allowing for an intuitive visual analysis. Finally, the script clears memory by deleting unnecessary variables to optimize performance.

This approach enables an unsupervised classification of Sentinel-2 imagery, allowing researchers to analyze environmental features without requiring labeled training data. The generated classification map provides valuable insights into surface variations, making it a useful tool for climate monitoring, environmental research, and land cover classification.

![image](https://github.com/user-attachments/assets/4dbda879-080b-406b-b1f0-24b97a1d7aa8)
This image represents the result of K-means clustering applied to a Sentinel-2 optical band (B4). The clustering algorithm groups pixels into two distinct clusters, as shown by the different colors. The yellow regions likely correspond to sea ice or land, while the darker regions represent open water or other surface types. The color bar on the right indicates cluster labels, with a no-data value (-1) assigned to areas outside the valid data range. This classification helps in distinguishing different surface features in remote sensing imagery.

#### GMM Implementation


```python
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```
This Python script performs Gaussian Mixture Model (GMM) clustering on a Sentinel-2 optical band (B4) image using the scikit-learn library. The code begins by defining the file paths for Sentinel-2 bands (B4, B3, and B2), with a specific base directory where the data is stored. It then reads the B4 band using the Rasterio library and stacks the data into a NumPy array. To ensure that only valid pixels are processed, a mask is created to exclude pixels with zero values.

Next, the script reshapes the valid band data into a format suitable for clustering and applies a Gaussian Mixture Model (GMM) with two components. GMM is a probabilistic clustering approach that models data as a mixture of multiple Gaussian distributions, allowing for soft clustering, meaning each pixel is assigned a probability of belonging to each cluster.

Once the model is trained, it assigns cluster labels to the valid pixels. The results are stored in a new array, with invalid areas being assigned a no-data value (-1). Finally, the script visualizes the clustering results using Matplotlib, where different clusters are color-coded, and a color bar is added to indicate cluster labels. This method is useful for analyzing sea ice, land cover, or other environmental patterns in Sentinel-2 imagery.

![2e7a5d5ebb2325493b5e9bdfa718c6d](https://github.com/user-attachments/assets/d6655404-0987-4593-8fc0-2bcc32270091)

Visualization of Gaussian Mixture Model (GMM) clustering applied to Sentinel-2 Band 4 imagery. The image showcases different clusters identified in the dataset, with the color scale representing distinct cluster labels. This method helps differentiate between various land cover types, such as sea ice, open water, and land surfaces, based on spectral reflectance patterns.


### Altimetry Classification
In this section, we explore the application of unsupervised learning techniques for classifying altimetry data, specifically focusing on distinguishing sea ice and leads within the Sentinel-3 altimetry dataset. This approach enables us to analyze surface features based on satellite-derived elevation measurements, improving our understanding of ice dynamics and oceanographic processes.

#### Read in Functions Needed
Before proceeding with modeling, it is essential to preprocess the data to ensure compatibility with analytical methods. This involves converting raw altimetry measurements into meaningful variables, such as peakiness and stack standard deviation (SSD), which provide insights into surface roughness and classification potential. Effective preprocessing enhances model accuracy and ensures a robust classification process.




```python
#
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy.ma as ma
import glob
from matplotlib.patches import Polygon
import scipy.spatial as spatial
from scipy.spatial import KDTree
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

#=========================================================================================================
#===================================  SUBFUNCTIONS  ======================================================
#=========================================================================================================

#*args and **kwargs allow you to pass an unspecified number of arguments to a function,
#so when writing the function definition, you do not need to know how many arguments will be passed to your function
#**kwargs allows you to pass keyworded variable length of arguments to a function.
#You should use **kwargs if you want to handle named arguments in a function.
#double star allows us to pass through keyword arguments (and any number of them).
def peakiness(waves, **kwargs):

    "finds peakiness of waveforms."

    #print("Beginning peakiness")
    # Kwargs are:
    #          wf_plots. specify a number n: wf_plots=n, to show the first n waveform plots. \

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import time

    print("Running peakiness function...")

    size=np.shape(waves)[0] #.shape property is a tuple of length .ndim containing the length of each dimensions
                            #Tuple of array dimensions.

    waves1=np.copy(waves)

    if waves1.ndim == 1: #number of array dimensions
        print('only one waveform in file')
        waves2=waves1.reshape(1,np.size(waves1)) #numpy.reshape(a, newshape, order='C'), a=array to be reshaped
        waves1=waves2

    # *args is used to send a non-keyworded variable length argument list to the function
    def by_row(waves, *args):
        "calculate peakiness for each waveform"
        maximum=np.nanmax(waves)
        if maximum > 0:

            maximum_bin=np.where(waves==maximum)
            #print(maximum_bin)
            maximum_bin=maximum_bin[0][0]
            waves_128=waves[maximum_bin-50:maximum_bin+78]

            waves=waves_128

            noise_floor=np.nanmean(waves[10:20])
            where_above_nf=np.where(waves > noise_floor)

            if np.shape(where_above_nf)[1] > 0:
                maximum=np.nanmax(waves[where_above_nf])
                total=np.sum(waves[where_above_nf])
                mean=np.nanmean(waves[where_above_nf])
                peaky=maximum/mean

            else:
                peaky = np.nan
                maximum = np.nan
                total = np.nan

        else:
            peaky = np.nan
            maximum = np.nan
            total = np.nan

        if 'maxs' in args:
            return maximum
        if 'totals' in args:
            return total
        if 'peaky' in args:
            return peaky

    peaky=np.apply_along_axis(by_row, 1, waves1, 'peaky') #numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)

    if 'wf_plots' in kwargs:
        maximums=np.apply_along_axis(by_row, 1, waves1, 'maxs')
        totals=np.apply_along_axis(by_row, 1, waves1, 'totals')

        for i in range(0,kwargs['wf_plots']):
            if i == 0:
                print("Plotting first "+str(kwargs['wf_plots'])+" waveforms")

            plt.plot(waves1[i,:])#, a, col[i],label=label[i])
            plt.axhline(maximums[i], color='green')
            plt.axvline(10, color='r')
            plt.axvline(19, color='r')
            plt.xlabel('Bin (of 256)')
            plt.ylabel('Power')
            plt.text(5,maximums[i],"maximum="+str(maximums[i]))
            plt.text(5,maximums[i]-2500,"total="+str(totals[i]))
            plt.text(5,maximums[i]-5000,"peakiness="+str(peaky[i]))
            plt.title('waveform '+str(i)+' of '+str(size)+'\n. Noise floor average taken between red lines.')
            plt.show()


    return peaky

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================


def unpack_gpod(variable):

    from scipy.interpolate import interp1d

    time_1hz=SAR_data.variables['time_01'][:]
    time_20hz=SAR_data.variables['time_20_ku'][:]
    time_20hzC = SAR_data.variables['time_20_c'][:]

    out=(SAR_data.variables[variable][:]).astype(float)  # convert from integer array to float.

    #if ma.is_masked(dataset.variables[variable][:]) == True:
    #print(variable,'is masked. Removing mask and replacing masked values with nan')
    out=np.ma.filled(out, np.nan)

    if len(out)==len(time_1hz):

        print(variable,'is 1hz. Expanding to 20hz...')
        out = interp1d(time_1hz,out,fill_value="extrapolate")(time_20hz)

    if len(out)==len(time_20hzC):
        print(variable, 'is c band, expanding to 20hz ku band dimension')
        out = interp1d(time_20hzC,out,fill_value="extrapolate")(time_20hz)
    return out


#=========================================================================================================
#=========================================================================================================
#=========================================================================================================

def calculate_SSD(RIP):

    from scipy.optimize import curve_fit
    # from scipy import asarray as ar,exp
    from numpy import asarray as ar, exp

    do_plot='Off'

    def gaussian(x,a,x0,sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    SSD=np.zeros(np.shape(RIP)[0])*np.nan
    x=np.arange(np.shape(RIP)[1])

    for i in range(np.shape(RIP)[0]):

        y=np.copy(RIP[i])
        y[(np.isnan(y)==True)]=0

        if 'popt' in locals():
            del(popt,pcov)

        SSD_calc=0.5*(np.sum(y**2)*np.sum(y**2)/np.sum(y**4))
        #print('SSD calculated from equation',SSD)

        #n = len(x)
        mean_est = sum(x * y) / sum(y)
        sigma_est = np.sqrt(sum(y * (x - mean_est)**2) / sum(y))
        #print('est. mean',mean,'est. sigma',sigma_est)

        try:
            popt,pcov = curve_fit(gaussian, x, y, p0=[max(y), mean_est, sigma_est],maxfev=10000)
        except RuntimeError as e:
            print("Gaussian SSD curve-fit error: "+str(e))
            #plt.plot(y)
            #plt.show()

        except TypeError as t:
            print("Gaussian SSD curve-fit error: "+str(t))

        if do_plot=='ON':

            plt.plot(x,y)
            plt.plot(x,gaussian(x,*popt),'ro:',label='fit')
            plt.axvline(popt[1])
            plt.axvspan(popt[1]-popt[2], popt[1]+popt[2], alpha=0.15, color='Navy')
            plt.show()

            print('popt',popt)
            print('curve fit SSD',popt[2])

        if 'popt' in locals():
            SSD[i]=abs(popt[2])


    return SSD
```
This Python script is designed for processing Sentinel-3 SAR altimetry data, specifically focusing on extracting and analyzing waveform characteristics such as peakiness and stack standard deviation (SSD). The code utilizes scientific and machine learning libraries, including NumPy, SciPy, Matplotlib, and Scikit-learn, to preprocess, analyze, and classify satellite altimetry data. The script begins by importing the necessary libraries for handling NetCDF data, performing mathematical operations, and implementing clustering techniques such as K-Means, DBSCAN, and Gaussian Mixture Models (GMMs).

One of the key functions, peakiness, calculates the peakiness of waveforms, which is a crucial metric for determining the sharpness of the altimeter return signal. This function identifies the maximum peak in the waveform, computes the peak-to-mean power ratio, and provides an option to visualize waveforms with annotated peak values. The script also includes the unpack_gpod function, which extracts altimetry variables from SAR data while handling masked values and converting low-resolution (1Hz) data to high-resolution (20Hz) timestamps using interpolation. This ensures consistency in data representation.

Another critical function, calculate_SSD, determines the Stack Standard Deviation (SSD), which measures variability in waveform returns. It applies Gaussian curve fitting to estimate signal spread, helping in the classification of surface types such as ice, water, and land. This function can also generate visualizations of SSD calculations, providing insights into signal characteristics.

Overall, this script is highly useful for scientific research and remote sensing applications, enabling the classification of sea ice, leads, and open ocean surfaces using waveform-derived features. Additionally, the inclusion of machine learning clustering techniques (K-Means, DBSCAN, GMMs) enhances data segmentation, making it possible to detect and categorize patterns in Sentinel-3 altimetry data efficiently.


```python
path = '/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/'
SAR_file = 'S3A_SR_2_LAN_SI_20190307T005808_20190307T012503_20230527T225016_1614_042_131______LN3_R_NT_005.SEN3'
SAR_data = Dataset(path + SAR_file + '/enhanced_measurement.nc')

SAR_lat = unpack_gpod('lat_20_ku')
SAR_lon = unpack_gpod('lon_20_ku')
waves   = unpack_gpod('waveform_20_ku')
sig_0   = unpack_gpod('sig0_water_20_ku')
RIP     = unpack_gpod('rip_20_ku')
flag = unpack_gpod('surf_type_class_20_ku')

# Filter out bad data points using criteria (here, lat >= -99999)
find = np.where(SAR_lat >= -99999)
SAR_lat = SAR_lat[find]
SAR_lon = SAR_lon[find]
waves   = waves[find]
sig_0   = sig_0[find]
RIP     = RIP[find]

# Calculate additional features
PP = peakiness(waves)
SSD = calculate_SSD(RIP)

# Convert to numpy arrays (if not already)
sig_0_np = np.array(sig_0)
PP_np    = np.array(PP)
SSD_np   = np.array(SSD)

# Create data matrix
data = np.column_stack((sig_0_np, PP_np, SSD_np))

# Standardize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
```





This Python script is designed to process and analyze Sentinel-3 SAR altimetry data, specifically using the SRAL (Synthetic Aperture Radar Altimeter) measurements. The script begins by defining the file path and loading the NetCDF dataset containing enhanced measurement data. It then extracts key parameters, including latitude (SAR_lat), longitude (SAR_lon), waveform data (waves), backscatter coefficient (sig_0), radar impulse response (RIP), and surface classification flags using the unpack_gpod function. These variables are crucial for understanding the surface characteristics of ocean, ice, and land regions.

To ensure data quality, the script applies a filtering step where only valid latitudes (≥ -99999) are retained, ensuring that invalid or missing data points are excluded from further analysis. Once the clean dataset is obtained, the script computes two essential waveform-derived features: Peakiness (PP) and Stack Standard Deviation (SSD). Peakiness (PP) is a measure of waveform sharpness, which helps differentiate between ocean, ice, and leads, while Stack Standard Deviation (SSD) quantifies variability in return signals, providing additional insights into surface roughness.

The extracted waveform parameters (sig_0, PP, and SSD) are then combined into a structured data matrix using np.column_stack(), ensuring that all relevant features are aligned for machine learning applications. To enhance the performance of clustering algorithms, the script applies feature scaling using the StandardScaler from Scikit-learn, which standardizes the data to have zero mean and unit variance. This step is crucial for ensuring that different features contribute equally to clustering models, such as K-Means, Gaussian Mixture Models (GMMs), or DBSCAN.

Overall, this script efficiently prepares and normalizes Sentinel-3 SAR altimetry data for further machine learning-based classification and clustering, enabling the identification of different surface types based on altimetric measurements.

There are some NaN values in the dataset so one way to deal with this is to delete them:
```python
# Remove any rows that contain NaN values
nan_count = np.isnan(data_normalized).sum()
print(f"Number of NaN values in the array: {nan_count}")

data_cleaned = data_normalized[~np.isnan(data_normalized).any(axis=1)]

mask = ~np.isnan(data_normalized).any(axis=1)
waves_cleaned = np.array(waves)[mask] 
flag_cleaned = np.array(flag)[mask]

data_cleaned = data_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
waves_cleaned = waves_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
flag_cleaned = flag_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
```

This script is responsible for data cleaning and preprocessing by handling missing values (NaNs) and filtering data based on predefined conditions.

First, the script counts the number of NaN (Not a Number) values in the data_normalized array using np.isnan(data_normalized).sum(). This step helps identify missing or invalid values, which may arise from incomplete satellite measurements or preprocessing errors. It then removes all rows containing NaN values using data_normalized[~np.isnan(data_normalized).any(axis=1)], ensuring that only complete and valid observations are used in further analysis.

Next, the script applies the same filtering process to related datasets, including waves_cleaned (which stores cleaned waveform data) and flag_cleaned (which contains classification flags). The mask ensures that only non-NaN entries from waves and flag are selected, keeping the datasets aligned.

Finally, the script filters the data based on surface classification flags, specifically keeping only values where flag_cleaned equals 1 or 2. These values typically correspond to specific surface types, such as open ocean, sea ice, or leads, depending on the dataset. This ensures that the dataset remains focused on relevant surface classifications, making it more suitable for unsupervised learning techniques like clustering or classification.

In summary, this script cleans the dataset, removes missing values, and retains only relevant data points based on surface classification flags, thereby improving the quality of input data for subsequent machine learning analyses.

Now, let’s proceed with running the GMM model as usual. Remember, you have the flexibility to substitute this with K-Means or any other preferred model:
```python
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_cleaned)
clusters_gmm = gmm.predict(data_cleaned)
```
This script applies Gaussian Mixture Model (GMM) clustering to the cleaned dataset, grouping data points into clusters based on their statistical properties.

First, a GaussianMixture model is initialized with two components (n_components=2), meaning that the algorithm will attempt to classify the data into two distinct clusters. The random_state=0 ensures that the results are reproducible by setting a fixed random seed.

Next, the model is trained using gmm.fit(data_cleaned), which fits the Gaussian Mixture Model to the preprocessed and standardized dataset (data_cleaned). The GMM algorithm assumes that the data is drawn from a mixture of two Gaussian distributions, each with its own mean and covariance. It iteratively refines the model using the Expectation-Maximization (EM) algorithm to estimate the parameters of these distributions.

Finally, the trained model is used to predict cluster assignments for each data point using gmm.predict(data_cleaned), storing the results in clusters_gmm. Each data point is assigned to one of the two Gaussian components, effectively segmenting the dataset into two distinct clusters.

This approach is particularly useful when working with continuous, normally distributed data and allows for soft clustering, meaning each data point has a probability of belonging to each cluster rather than a hard assignment. This makes GMM more flexible than K-means, especially when clusters have different shapes and variances.

We can also inspect how many data points are there in each class of your clustering prediction:

```python
unique, counts = np.unique(clusters_gmm, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Cluster counts:", class_counts)
```

This script analyzes the distribution of data points across the clusters identified by the Gaussian Mixture Model (GMM).

First, the np.unique() function is used to count the occurrences of each cluster label in clusters_gmm, which stores the cluster assignments for the dataset. The return_counts=True argument ensures that the function returns both the unique cluster labels and their corresponding counts.

Next, the results are combined into a dictionary using dict(zip(unique, counts)), where the cluster labels serve as keys, and their respective counts as values. This structured output is stored in class_counts.

Finally, the script prints the number of data points assigned to each cluster, giving insight into how the Gaussian Mixture Model has distributed the data. If the clusters are significantly imbalanced, this may indicate differences in the density or distribution of features, which could warrant further analysis or tuning of the clustering parameters.

```python
# mean and standard deviation for all echoes
mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)


mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

plt.plot(mean_lead, label='lead')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

plt.title('Plot of mean and standard deviation for each class')
plt.legend()
```
This script performs an analysis and visualization of the mean and standard deviation of waveforms for different clusters identified by the Gaussian Mixture Model (GMM), aiming to differentiate between sea ice and leads in the dataset. First, the script calculates the mean and standard deviation for waveforms categorized as sea ice (cluster 0). It extracts the relevant waveforms using waves_cleaned[clusters_gmm==0] and computes their mean and standard deviation using NumPy functions. The mean waveform is then plotted, with a shaded area representing one standard deviation to indicate variability.

Next, the same process is repeated for waveforms categorized as leads (cluster 1), ensuring a comparative analysis between the two classes. The script then plots both the mean waveforms for ice and leads while shading their respective standard deviations for better visualization. A legend is included to clearly differentiate the two categories, and the plot is titled "Plot of mean and standard deviation for each class" to indicate the purpose of the visualization.

This approach allows for an intuitive understanding of waveform differences between sea ice and leads, highlighting variations in their reflectivity patterns based on the statistical properties of the dataset. Such an analysis is crucial for effective classification and interpretation of altimetry data in remote sensing applications.

![7c3651827a7ebef8494864dcce54cc8](https://github.com/user-attachments/assets/88e34ef0-0769-4cbb-9a76-073679057a75)

Visualization of the mean waveforms for sea ice and leads, with shaded areas representing one standard deviation. This plot helps to compare the reflectivity patterns of the two surface types, highlighting differences in waveform characteristics. The blue curve corresponds to sea ice, while the orange curve represents leads, demonstrating variability in their respective waveform signals.



```python
x = np.stack([np.arange(1,waves_cleaned.shape[1]+1)]*waves_cleaned.shape[0])
plt.plot(x,waves_cleaned)  # plot of all the echos
plt.show()
```

This code generates a plot of all waveform echoes in the dataset. It first creates a stacked array x, which represents the bin indices for the waveforms. The waveforms stored in waves_cleaned are then plotted against these bin indices using plt.plot(x, waves_cleaned). This visualization helps in understanding the distribution and variability of waveform signals across different observations. Finally, plt.show() displays the plot, allowing for an overall inspection of the waveform structures.

![image](https://github.com/user-attachments/assets/f0985c42-840f-49b7-a879-13fbe593d8e2)

Visualization of all waveform echoes, showing the distribution and intensity of the signals across different bins. The central peak indicates the dominant signal response, while the spread represents variations in waveform characteristics.

```python
# plot echos for the lead cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==1].shape[1]+1)]*waves_cleaned[clusters_gmm==1].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==1])  # plot of all the echos
plt.show()
```

This code generates a plot displaying all waveform echoes that belong to the lead cluster, as identified by the Gaussian Mixture Model (GMM). It first creates an array x representing the bin positions for each waveform. Then, it extracts only the waveforms assigned to the lead class (clusters_gmm==1) and plots them to visualize their distribution. The resulting plot helps analyze the waveform characteristics specific to the lead category.

![image](https://github.com/user-attachments/assets/42acd430-03e3-406f-b61f-d0bbdee17288)


Visualization of all waveform echoes classified as lead by the Gaussian Mixture Model (GMM). The plot represents the waveform intensity distribution across different bins, highlighting the distinct characteristics of the lead category in the Sentinel-3 altimetry dataset.

```python
# plot echos for the sea ice cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==0].shape[1]+1)]*waves_cleaned[clusters_gmm==0].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==0])  # plot of all the echos
plt.show()
```

This code generates a plot of waveforms associated with the sea ice cluster, which was identified using the Gaussian Mixture Model (GMM) clustering technique. It first creates an array x that represents the range of waveform bins for each cleaned echo belonging to the sea ice cluster. Then, it plots all the waveforms corresponding to this cluster, allowing for a visual representation of their structural characteristics. The plt.show() function displays the resulting plot, helping to analyze the waveform patterns of sea ice echoes in the Sentinel-3 altimetry data.


![image](https://github.com/user-attachments/assets/21330790-9b6e-4de3-b764-2c4f555f22c6)


Visualization of waveforms classified as sea ice using the Gaussian Mixture Model (GMM). Each line represents an individual waveform, providing insight into the distinct structural characteristics of sea ice echoes in Sentinel-3 altimetry data.


### Scatter Plots of Clustered Data

This code visualizes the clustering results using scatter plots, where different colors represent different clusters (clusters_gmm):

```python
plt.scatter(data_cleaned[:,0],data_cleaned[:,1],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("PP")
plt.show()
plt.scatter(data_cleaned[:,0],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("SSD")
plt.show()
plt.scatter(data_cleaned[:,1],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("PP")
plt.ylabel("SSD")
```

This code generates three scatter plots to visualize the clustering results obtained using the Gaussian Mixture Model (GMM) on Sentinel-3 altimetry data. Each scatter plot represents the relationship between different extracted features: sigma naught (σ₀), Peakiness Parameter (PP), and Stack Standard Deviation (SSD). The first plot visualizes σ₀ vs. PP, showing how these two parameters are distributed and how the clustering algorithm has grouped the data points. The second plot illustrates σ₀ vs. SSD, providing insights into how the backscatter coefficient varies with waveform deviation. The third plot displays PP vs. SSD, helping to further distinguish between different surface types, such as sea ice and leads. Each data point in the scatter plots is color-coded according to its assigned cluster, allowing for a clear visual representation of the clustering results. These plots help in understanding the distinct characteristics of sea ice and leads based on their altimetric properties.


![image](https://github.com/user-attachments/assets/96752147-485b-4edb-8cd7-4f6cfe140ed3)


![image](https://github.com/user-attachments/assets/cba6ea58-3e8a-44b9-bc7e-34beabcb75f2)


![image](https://github.com/user-attachments/assets/b6875888-69e5-40cb-96e3-b4743761c005)


These scatter plots visualize the clustering results of the Gaussian Mixture Model (GMM) applied to the Sentinel-3 altimetry dataset. Each plot highlights the relationships between different feature pairs:

First Plot: Displays the relationship between sig_0 (backscatter coefficient) and PP (peakiness parameter). The clusters, represented by different colors, show distinct groupings.

Second Plot: Shows the correlation between sig_0 and SSD (Stack Standard Deviation), indicating how these variables contribute to cluster separation.

Third Plot: Represents the distribution of clusters in the PP vs. SSD feature space, further illustrating the separation between sea ice and leads.

These visualizations help in understanding the structure and classification of different surface types using GMM clustering.


### Waveform Alignment Using Cross-Correlation

This code aligns waveforms in the cluster where clusters_gmm == 0 by using cross-correlation:

```python
from scipy.signal import correlate
 
# Find the reference point (e.g., the peak)
reference_point_index = np.argmax(np.mean(waves_cleaned[clusters_gmm==0], axis=0))
 
# Calculate cross-correlation with the reference point
aligned_waves = []
for wave in waves_cleaned[clusters_gmm==0][::len(waves_cleaned[clusters_gmm == 0]) // 10]:
    correlation = correlate(wave, waves_cleaned[clusters_gmm==0][0])
    shift = len(wave) - np.argmax(correlation)
    aligned_wave = np.roll(wave, shift)
    aligned_waves.append(aligned_wave)
 
# Plot aligned waves
for aligned_wave in aligned_waves:
    plt.plot(aligned_wave)
 
plt.title('Plot of 10 equally spaced functions where clusters_gmm = 0 (aligned)')
```

This code snippet performs cross-correlation-based alignment of waveforms within the cluster labeled as 0 by the Gaussian Mixture Model (GMM). The process starts by identifying a reference point, specifically the peak of the mean waveform for the cluster. Next, it computes the cross-correlation between each waveform and the first waveform in the cluster, determining the shift required to align the peaks. The waveforms are then adjusted accordingly using np.roll(), ensuring they are aligned with the reference point. Finally, a subset of ten equally spaced waveforms is plotted to visualize the alignment, providing insight into the consistency of waveform structures within the identified cluster.



![image](https://github.com/user-attachments/assets/a27f9aef-6c03-40db-80ef-5e370ef71738)


Caption: Visualization of 10 equally spaced waveforms from the cluster labeled as sea ice (clusters_gmm = 0) after alignment using cross-correlation. This alignment ensures that the peaks of the waveforms are synchronized, allowing for a clearer comparison of waveform structures within the identified cluster.

### Compare with ESA data
In the ESA dataset, sea ice = 1 and lead = 2. Therefore, we need to subtract 1 from it so our predicted labels are comparable with the official product labels:

```python
flag_cleaned_modified = flag_cleaned - 1
```

This line of code, flag_cleaned_modified = flag_cleaned - 1, modifies the flag_cleaned array by subtracting 1 from each of its elements. The purpose of this operation is likely to adjust the labeling of data categories, ensuring that the values start from zero instead of one. This can be useful for compatibility with machine learning models that expect zero-based indexing or for standardizing the dataset before further processing. The modified array, flag_cleaned_modified, retains the same structure as flag_cleaned but with all values shifted down by one.


```python
from sklearn.metrics import confusion_matrix, classification_report

true_labels = flag_cleaned_modified   # true labels from the ESA dataset
predicted_gmm = clusters_gmm          # predicted labels from GMM method

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_gmm)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Compute classification report
class_report = classification_report(true_labels, predicted_gmm)

# Print classification report
print("\nClassification Report:")
print(class_report)
```

This code evaluates the performance of the Gaussian Mixture Model (GMM) clustering method by comparing its predicted labels to the actual labels from the ESA dataset. First, the true labels are extracted from flag_cleaned_modified, while the predicted labels are obtained from the GMM clustering results stored in clusters_gmm. To assess the accuracy of the clustering, the code computes a confusion matrix using confusion_matrix(true_labels, predicted_gmm), which provides a summary of correct and incorrect classifications for each category. The confusion matrix is then printed to analyze the distribution of misclassified data points. Additionally, the code generates a classification report using classification_report(true_labels, predicted_gmm), which calculates key performance metrics such as precision, recall, and F1-score for each class. This report provides a more detailed evaluation of the clustering model's effectiveness in distinguishing between different data categories. By analyzing these outputs, we can determine how well the GMM model aligns with the actual classifications and gain insights into potential improvements.

```python
Confusion Matrix:
[[8856   22]
 [  24 3293]]

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      8878
         1.0       0.99      0.99      0.99      3317

    accuracy                           1.00     12195
   macro avg       1.00      1.00      1.00     12195
weighted avg       1.00      1.00      1.00     12195
```


The confusion matrix and classification report summarize the performance of the Gaussian Mixture Model (GMM) clustering. The confusion matrix shows that 8,856 instances were correctly classified as class 0, while 3,293 instances were correctly classified as class 1. There were 22 misclassified instances of class 0 and 24 misclassified instances of class 1. The classification report indicates a precision, recall, and F1-score close to 1.00, demonstrating high accuracy in distinguishing between the two classes. The overall accuracy is 100%, suggesting that the clustering method effectively separates the data into meaningful groups.













