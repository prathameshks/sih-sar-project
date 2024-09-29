import ee
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import geemap
from sar_change_detection.settings import env, BASE_DIR
import os
import json
from datetime import datetime
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from segmentation_models_pytorch import DeepLabV3Plus
from PIL import Image
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from django.template import loader
from django.http import HttpResponse
import json

# Initialize Earth Engine
service_account = env('GCP_SERVICE_ACCOUNT_ID')
credentials = ee.ServiceAccountCredentials(service_account, os.path.join(BASE_DIR, 'service-account-key.json'))
ee.Initialize(credentials)

# Load pre-trained models
def load_cnn_model():
    model = DeepLabV3Plus(encoder_name="resnet34", classes=1)
    # model.load_state_dict(torch.load('path_to_your_pretrained_deeplabv3plus_model.pth'))
    model.eval()
    return model

def load_random_forest_model():
    # Load your pre-trained Random Forest model here
    # If you don't have a pre-trained model, we'll create a new one
    return RandomForestClassifier(n_estimators=100, random_state=42)

cnn_model = load_cnn_model()
rf_model = load_random_forest_model()

# Preprocessing function for CNN input
def preprocess_for_cnn(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

@csrf_exempt
def detect_changes(request):
    if request.method == 'POST':
        # try:
        if True:
            # Parse and validate input parameters
            aoi = request.POST.get('aoi') 
            start_date = request.POST.get('start_date')
            end_date = request.POST.get('end_date')
            
            change_type = request.POST.get('change_type', 'all')
            size_threshold = request.POST.get('size_threshold',1000)
            output_type = request.POST.get('output_type','outline')
            file_type = request.POST.get('file_type','geojson')
            
            print(aoi, start_date, end_date)
            print(change_type, size_threshold, output_type, file_type)

            # Validate parameters
            if not all([aoi, start_date, end_date]):
                return JsonResponse({'error': 'Missing required parameters'}, status=400)

            # Convert AOI to ee.Geometry
            aoi = json.loads(aoi)
            aoi_geo = geemap.geojson_to_ee(aoi)

            # Load Sentinel-1 and Sentinel-2 imagery
            s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(aoi_geo) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))

            s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(aoi_geo) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

            # Perform change detection using CNN
            def cnn_change_detection(image_pair):
                # Download the image pair from Earth Engine
                image_data = image_pair.getInfo()
                
                # Preprocess the image for the CNN
                input_tensor = preprocess_for_cnn(Image.fromarray(np.array(image_data['bands'][0]['data'], dtype=np.uint8).reshape(image_data['bands'][0]['dimensions'])))
                
                # Run it through the CNN
                with torch.no_grad():
                    output = cnn_model(input_tensor)
                    change_mask = (output > 0.5).float()  # Threshold the output
                
                # Convert the PyTorch tensor to a numpy array
                change_mask_np = change_mask.squeeze().numpy()
                
                # Create an Earth Engine image from the numpy array
                change_mask_ee = ee.Image.fromArray(change_mask_np).reproject(crs=image_pair.projection())
                
                return change_mask_ee
            
            changes = s1_collection.map(cnn_change_detection)

            # Apply DBSCAN clustering to group changes
            def apply_dbscan(image):
                # Extract features for clustering
                features = image.select(['VV', 'VH']).sample(region=aoi_geo, scale=30, numPixels=5000)
                feature_array = np.array(features.getInfo()['features'])
                
                # Perform DBSCAN clustering
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = dbscan.fit_predict(feature_array)
                
                # Create a new feature collection with cluster labels
                clustered_features = ee.FeatureCollection(
                    [ee.Feature(ee.Geometry.Point([f['geometry']['coordinates'][0], f['geometry']['coordinates'][1]]), 
                                {'cluster': int(label)}) for f, label in zip(features.getInfo()['features'], cluster_labels)]
                )
                
                # Convert clustered features to an image
                clustered_image = ee.Image().int().paint(clustered_features, 'cluster')
                
                return image.addBands(clustered_image.rename('cluster'))

            clustered_changes = changes.map(apply_dbscan)

            # Extract features for Random Forest classification
            def extract_features(image):
                # Extract relevant features for classification
                return image.select(['VV', 'VH', 'change', 'cluster'])

            features = clustered_changes.map(extract_features)

            # Apply Random Forest classification
            def apply_random_forest(features):
                # Download features from Earth Engine
                feature_array = np.array(features.getInfo()['features'])
                X = feature_array[:, :4]  # VV, VH, change, cluster
                y = feature_array[:, 4]   # Assuming the 5th column is the label (you may need to adjust this)
                
                # Train the Random Forest model (or use a pre-trained model)
                rf_model.fit(X, y)
                
                # Make predictions
                predictions = rf_model.predict(X)
                
                # Create a new feature collection with predictions
                predicted_features = ee.FeatureCollection(
                    [ee.Feature(ee.Geometry.Point([f['geometry']['coordinates'][0], f['geometry']['coordinates'][1]]), 
                                {'prediction': int(pred)}) for f, pred in zip(features.getInfo()['features'], predictions)]
                )
                
                # Convert predicted features to an image
                prediction_image = ee.Image().int().paint(predicted_features, 'prediction')
                
                return features.addBands(prediction_image.rename('classification'))

            classified_changes = features.map(apply_random_forest)

            # Filter changes based on type and size
            if change_type != 'all':
                classified_changes = classified_changes.select(change_type)
            
            size_filtered_changes = classified_changes.updateMask(
                classified_changes.connectedPixelCount().gte(size_threshold / 900)  # Assuming 30m resolution
            )

            # Generate output based on specified type
            if output_type == 'contrast':
                output = size_filtered_changes.visualize(min=0, max=1, palette=['blue', 'red'])
            elif output_type == 'outline':
                output = size_filtered_changes.reduceToVectors(scale=30, geometryType='polygon')
            else:  # heatmap
                output = size_filtered_changes.focalMax(2).kernelGaussian(5)

            # Get the result as GeoJSON or Shapefile
            if file_type == 'geojson':
                result = geemap.ee_to_geojson(output)
            else:  # shapefile
                result = geemap.ee_to_shapefile(output, filename='changes.shp')

            # Prepare the response
            response = {
                'changes': result,
                'metadata': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'change_type': change_type,
                    'size_threshold': size_threshold,
                    'output_type': output_type,
                    'file_type': file_type
                }
            }

            return JsonResponse(response)

        # except Exception as e:
        #     # print detailed error message
        #     print(e)
        #     return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def index(request):
    template = loader.get_template('change_detection.html')
    context = {}
    return HttpResponse(template.render(context, request))