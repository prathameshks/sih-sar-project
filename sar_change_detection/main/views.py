import ee
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import geemap
from sar_change_detection.settings import env,BASE_DIR
import os
from django.http import HttpResponse
from django.template import loader
import json

# Initialize Earth Engine
service_account = env('GCP_SERVICE_ACCOUNT_ID')
credentials = ee.ServiceAccountCredentials(service_account,os.path.join(BASE_DIR, 'service-account-key.json'))
ee.Initialize(credentials)

@csrf_exempt
def detect_changes(request):
    if request.method == 'POST':
        # Parse input parameters from request
        aoi = request.POST.get('aoi')  # Assuming AOI is sent as a GeoJSON string
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        
        # print(request.POST)
        
        # convert aoi from string to dict
        aoi = json.loads(aoi)

        # Convert AOI to ee.Geometry
        aoi_geo = geemap.geojson_to_ee(aoi)
        
        # Load Sentinel-2 imagery
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi_geo) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        # print(collection.size())
        # Ensure we have images in the collection
        if collection.size().getInfo() == 0:
            raise ValueError("No images found for the specified date range and area.")
        
        # Perform change detection (example using NDVI)
        def calculate_ndvi(image):
            return image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        ndvi_collection = collection.map(calculate_ndvi)
        
        # Get the first and last images
        initial_ndvi = ee.Image(ndvi_collection.first())
        final_ndvi = ee.Image(ndvi_collection.sort('system:time_start', False).first())
        
        # Ensure both images exist before subtraction
        ndvi_difference = final_ndvi.subtract(initial_ndvi)
        
        # Threshold for change detection
        threshold = 0.2
        changes = ndvi_difference.abs().gt(threshold)
        
        # Convert changes to vector format
        vectors = changes.reduceToVectors(
            geometry=aoi_geo,
            scale=10,
            eightConnected=False
        )
        
        # Get the result as GeoJSON
        geojson = geemap.ee_to_geojson(vectors)
        
        return JsonResponse({'changes': geojson})
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def index(request):
    template = loader.get_template('change_detection.html')
    context = {}
    return HttpResponse(template.render(context, request))