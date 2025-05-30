import os
import numpy as np
from streamlit_drawable_canvas import st_canvas
import torch
import open_clip
import pandas as pd
import requests
import medical_assistant_package.config as cfg

model, _, preprocess = open_clip.create_model_and_transforms(cfg.OPENCLIP_MODEL_TYPE, pretrained='openai')
model.eval()
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")

medicine_df = pd.read_csv(cfg.MEDICINE_CSV)

# collect column name
medicine_df_column_list = medicine_df.columns.to_list()[0:] 
print(medicine_df_column_list)

# This dataset contains the url of medicine image.
# So, I follow the "download-met-museum-data.py" from week 3
def download_medicine_image(medicine_name,save_directory,img_url):
    
    os.makedirs(save_directory, exist_ok=True)
    url = img_url  


    response = requests.get(url,stream=True)
    
    try:

        if response.status_code == 200:
        
            img_name = medicine_name + '.jpg'
            save_path = os.path.join(save_directory, img_name)
     
            with open(save_path, 'wb') as file:
               
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Image successfully downloaded: {save_path}")
        else:
            print(f"Failed to download image from {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while downloading the image: {e}")

medicine_name_df = medicine_df['Medicine Name']
medicine_name_list = medicine_name_df.to_list()
medicine_url_list = medicine_df['Image URL'].to_list()
# print(medicine_name_list)

for name,url in zip(medicine_name_list, medicine_url_list):
    download_medicine_image(name,"Medicine_Picture",url)

def get_clip_embedding_from_PIL_image(image):
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor).squeeze().numpy()
    return embedding

medicine_use_df = medicine_df['Uses'].to_list

