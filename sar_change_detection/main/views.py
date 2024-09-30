import ee
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import geemap
from sar_change_detection.settings import env, BASE_DIR
import os
import json
# from datetime import datetime
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.cluster import DBSCAN
# from segmentation_models_pytorch import DeepLabV3Plus
from PIL import Image
# import rasterio
# from rasterio.features import shapes
# from shapely.geometry import shape
# import geopandas as gpd
from django.template import loader
from django.http import HttpResponse
import json
import geojson

# Initialize Earth Engine
service_account = env('GCP_SERVICE_ACCOUNT_ID')
credentials = ee.ServiceAccountCredentials(service_account, os.path.join(BASE_DIR, 'service-account-key.json'))
ee.Initialize(credentials)

def get_closest_images(s2_collection, start_date, end_date):
    """
    Gets the closest non-None Sentinel-2 images BEFORE the specified start and end dates.

    Args:
        s2_collection: An ee.ImageCollection of Sentinel-2 images.
        start_date: The start date of the image search.
        end_date: The end date of the image search.

    Returns:
        A list containing the closest non-None Sentinel-2 images BEFORE the start and end dates.
    """

    start_images = s2_collection.filterDate('1900-01-01', start_date).sort('system:time_start', False)
    end_images = s2_collection.filterDate('1900-01-01', end_date).sort('system:time_start', False)

    closest_start_image = None
    closest_end_image = None

    # Iterate to find the first non-None image for the start date
    start_list = start_images.toList(start_images.size()) # convert the collection to a list 
    for i in range(start_images.size().getInfo()):
        img = ee.Image(start_list.get(i)) # get the image from the list using index i
        if img.getInfo() is not None:
            closest_start_image = img
            break

    # Iterate to find the first non-None image for the end date
    end_list = end_images.toList(end_images.size())
    for i in range(end_images.size().getInfo()):
        img = ee.Image(end_list.get(i))
        if img.getInfo() is not None:
            closest_end_image = img
            break

    return [closest_start_image, closest_end_image]

@csrf_exempt
def detect_changes(request):
    if request.method == 'POST':
        # try :
        if True:
            # Parse input parameters
            aoi = json.loads(request.POST.get('aoi'))
            start_date = request.POST.get('start_date')
            end_date = request.POST.get('end_date')
            threshold_val = float(request.POST.get('threshold_range'))
            if(threshold_val<0.1): threshold_val=0.1
            else: threshold_val = 0.7
            
            # aoi_geo = geemap.geojson_to_ee(aoi)
            aoi_geo = geemap.geojson_to_ee(aoi).geometry()  # Convert to Geometry


            s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(aoi_geo) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

            # Get the closest images
            # start_date = ee.Date('2024-09-01')  # Replace with your start date
            # end_date = ee.Date('2024-09-28')  # Replace with your end date
            images = get_closest_images(s2_collection, start_date, end_date)

            # Access the images (these won't be None even if exact dates aren't available)
            s2_start = images[0]
            s2_end = images[1]

            start_image_date = s2_start.date().format('YYYY-MM-dd HH:mm:ss').getInfo()
            end_image_date = s2_end.date().format('YYYY-MM-dd HH:mm:ss').getInfo()

            # Calculate NDVI for both images
            ndvi1 = s2_start.normalizedDifference(['B8', 'B4'])
            ndvi2 = s2_end.normalizedDifference(['B8', 'B4'])

            # Compute NDVI difference
            ndvi_diff = ndvi2.subtract(ndvi1)

            # Threshold the difference image
            threshold = ndvi_diff.abs().gt(threshold_val)  # Adjust threshold as needed

            # Apply morphological operations to remove noise
            kernel = ee.Kernel.circle(radius=1)
            significant_changes = threshold.focal_min(kernel=kernel).focal_max(kernel=kernel)

            # Print some information to check if significant_changes is empty
            # print('Significant changes count:', significant_changes.bandNames().size().getInfo()) 
            
            # Convert changes to vectors
            vectors = significant_changes.reduceToVectors(
                geometry=aoi_geo,
                scale=10,
                geometryType='polygon',
                maxPixels=1e9,
                eightConnected=False
            )

            # Prepare outputs
            start_image_url = s2_start.getThumbURL({
                'min': 0,
                'max': 3000,
                'dimensions': 1024,
                'region': aoi_geo,
                'bands': ['B4', 'B3', 'B2']
            })

            end_image_url = s2_end.getThumbURL({
                'min': 0,
                'max': 3000,
                'dimensions': 1024,
                'region': aoi_geo,
                'bands': ['B4', 'B3', 'B2']
            })

            change_image_url = significant_changes.getThumbURL({
                'min': 0,
                'max': 1,
                'dimensions': 1024,
                'region': aoi_geo,
                'palette': ['black', 'white']
            })

            # Get GeoJSON of changes
            # geojson_from_vector = geemap.ee_to_geojson(vectors)
            
            # Convert to GeoJSON
            geojson_data = vectors.getInfo()
            geojson_string = json.dumps(geojson_data)
            
            # get bounds of AOI
            bounds = aoi_geo.bounds().getInfo()

            # Save as GeoJSON file for download
            # with open('change_detection.geojson', 'w') as f:
            #     geojson.dump(geojson_data, f)

            # Prepare the response
            response = {
                'start_image': start_image_url,
                'end_image': end_image_url,
                'change_image': change_image_url,
                'changes_geojson': geojson_data,
                'geojson_data': geojson_string,
                'start_date': start_image_date,
                'end_date': end_image_date,
                'are_images_available': start_image_date != end_image_date,
                'bounds': bounds
            }

            return JsonResponse(response)
        try:
            pass
        except Exception as e:
            print(e)
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def detect_changes_dummy(request):
    response = {
'start_image' : 'https://earthengine.googleapis.com/v1/projects/earthengine-legacy/thumbnails/d9ca3650cc9bb81f8dfa00e199488d84-75c293210111bdebdc76b67a5f40f8a2:getPixels',
'end_image' : 'https://earthengine.googleapis.com/v1/projects/earthengine-legacy/thumbnails/3bbec56d91c3f3acb5b4883bde516121-21b2ef1a6e0f5f87181965f3ee10e8ff:getPixels',
'change_image' : 'https://earthengine.googleapis.com/v1/projects/earthengine-legacy/thumbnails/938736a42876bfcd98b4216344922204-458b48be4ba055f051bca5b9b3b07936:getPixels',
'changes_geojson' : {'type': 'FeatureCollection', 'columns': {'count': 'Long<0, 4294967295>', 'label': 'Byte<0, 1>', 'system:index': 'String'}, 'features': [{'type': 'Feature', 'geometry': {'geodesic': False, 'type': 'Polygon', 'coordinates': [[[82.13436931633424, 22.189324478416093], [82.13446630616885, 22.18932380290995], [82.13445760192494, 22.188239893352513], [82.14182876238527, 22.18818838863434], [82.14182949242813, 22.18827871420921], [82.15424403729492, 22.18819121539898], [82.15424477527888, 22.18828154057859], [82.16675625384454, 22.188192400941592], [82.16675699982352, 22.18828272571696], [82.17926842252295, 22.18819262513414], [82.17926917650591, 22.188282949505265], [82.19178054266538, 22.1881918879856], [82.19178130464336, 22.188282211943502], [82.20419562693456, 22.188190906540218], [82.20419639685363, 22.188281230084893], [82.21670764838808, 22.188188254165528], [82.21670842629318, 22.18827857729698], [82.22223663608479, 22.188237188767737], [82.22232496514495, 22.1984436746392], [82.22222797211235, 22.198444402741703], [82.22232808034964, 22.21000571645082], [82.22223107938491, 22.210006444912647], [82.22233124557253, 22.221567742191578], [82.22223423664873, 22.221568471012734], [82.22226241802989, 22.22482008295796], [82.21207622425626, 22.22489629983552], [82.21207544788729, 22.224805976964582], [82.19956092908134, 22.22489874169901], [82.19956016072534, 22.22480841840586], [82.1870455842206, 22.2249002204697], [82.18704482388655, 22.224809896763325], [82.17462720327806, 22.224900035668277], [82.17462645089411, 22.22480971155766], [82.162111761543, 22.224899595709385], [82.16211101718099, 22.224809271185546], [82.14969328595397, 22.224897507027535], [82.14969254954205, 22.224807182108435], [82.13717775006648, 22.224895148303126], [82.13717702166754, 22.22480482298877], [82.1344606245762, 22.224823789236144], [82.1343879775407, 22.215791244451445], [82.13448498556617, 22.215790568163765], [82.13437897887395, 22.202603034825817], [82.13447597783544, 22.202602358933397], [82.13436931633424, 22.189324478416093]]]}, 'id': '+1770+4591', 'properties': {'count': 367129, 'label': 1}}]},
'start_date' : '2024-05-24 05:12:40',
'end_date' : '2024-05-04 05:12:44',
'are_images_available' : 'True',
    }

    return JsonResponse(response)
    
def index(request):    
    template = loader.get_template('change_detection.html')
    context = {}
    return HttpResponse(template.render(context, request))