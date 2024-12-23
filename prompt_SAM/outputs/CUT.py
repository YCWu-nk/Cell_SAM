from PIL import Image  
import os  
  

input_dir = 'C:/Users/Administrator/Desktop/pin/'  
output_dir = 'C:/Users/Administrator/Desktop/pin/'  
if not os.path.exists(output_dir):  
    os.makedirs(output_dir)  
  

for filename in os.listdir(input_dir):  
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):  
        img_path = os.path.join(input_dir, filename)  
        img = Image.open(img_path)  
          

        left = 128
        top = 120
        right = 898 
        bottom = 890
          

        cropped_img = img.crop((left, top, right, bottom))  
          

        output_path = os.path.join(output_dir, filename)  
        cropped_img.save(output_path)  
  
print("DONE！")