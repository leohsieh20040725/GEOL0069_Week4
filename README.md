# GEOL0069_Week4
This repository contains my Week 4 assignment, which classifies echoes in leads and sea ice using machine learning. It generates an average echo shape, computes the standard deviation for both categories, and compares results against the ESA official classification using a confusion matrix

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
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>


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
        <li><a href="#Proceeding with Sentinel-3 OLCI Download">Proceeding with Sentinel-3 OLCI Download</a></li>
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
        <li><a href="#Introduction to Gaussian Mixture Models</a></li>
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
       
       </ul>
    </li>
<li><a href="#Unsupervised Learning">Unsupervised Learning</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>





<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

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
## Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)
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

### Co-locate the data
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

