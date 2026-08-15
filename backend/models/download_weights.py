import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def download_yolo_weights():
    """Forces Ultralytics to download yolov8n.pt weights."""
    print("[Setup] Fetching YOLOv8 weights...")
    try:
        model = YOLO("yolov8n.pt")
        print("[Setup] yolov8n.pt downloaded successfully.")
    except Exception as e:
        print(f"[Setup] Error downloading YOLO weights: {e}")

def create_synthetic_plate(text, filename, blur=False, skew=False, noise=False):
    """
    Creates a synthetic Indian license plate image using OpenCV and PIL.
    """
    # Standard dimensions: 340x80 pixels (typical plate ratio)
    width, height = 340, 80
    
    # Create white background image
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw border
    cv2.rectangle(img, (2, 2), (width-3, height-3), (0, 0, 0), 2)
    cv2.rectangle(img, (5, 5), (width-6, height-6), (0, 0, 0), 1)
    
    # Convert to PIL to draw clean text
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a nice font, otherwise fallback to default
    font = None
    font_paths = [
        "C:\\Windows\\Fonts\\arialbd.ttf",  # Windows Arial Bold
        "C:\\Windows\\Fonts\\consolab.ttf", # Windows Consolas Bold
        "C:\\Windows\\Fonts\\tahomabd.ttf"  # Windows Tahoma Bold
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 44)
                break
            except Exception:
                continue
                
    if font is None:
        font = ImageFont.load_default()
        
    # Calculate text size using textbbox to center it
    text_str = f" {text[0:2]} {text[2:4]} {text[4:6]} {text[6:]} "
    bbox = draw.textbbox((0, 0), text_str, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    tx = (width - text_w) // 2
    ty = (height - text_h) // 2 - 5  # Adjust slight vertical offset
    
    draw.text((tx, ty), text_str, fill=(0, 0, 0), font=font)
    
    # Convert back to OpenCV BGR
    img = np.array(pil_img)
    
    # Apply modifications
    if blur:
        # Gaussian Blur
        img = cv2.GaussianBlur(img, (7, 7), 0)
        
    if noise:
        # Add salt-and-pepper noise
        row, col, ch = img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss * 50
        img = np.clip(noisy, 0, 255).astype(np.uint8)
        
    if skew:
        # Rotate image slightly (e.g., 6 degrees)
        angle = 6.0
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
    # Save image
    cv2.imwrite(filename, img)
    print(f"[Setup] Created synthetic plate: {filename}")

def create_synthetic_traffic_scene(filename):
    """
    Creates a simulated traffic image.
    Contains a background road, a drawn motorcycle shape, a rider with head, and a plate.
    This serves as a mock input image to test the detection pipeline.
    """
    width, height = 640, 480
    
    # Draw a simple background: gray road, green grass, blue sky
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[0:150, :] = [235, 206, 135] # Sky (BGR: light blue)
    img[150:280, :] = [100, 180, 100] # Grass (BGR: green)
    img[280:, :] = [80, 80, 80]     # Road (BGR: gray)
    
    # Draw road lanes (yellow dashed line)
    cv2.line(img, (width//2, 320), (width//2, 340), (0, 255, 255), 3)
    cv2.line(img, (width//2, 380), (width//2, 410), (0, 255, 255), 3)
    cv2.line(img, (width//2, 450), (width//2, 480), (0, 255, 255), 3)
    
    # Draw a simulated rider on a motorcycle (coordinates around center)
    # 1. Motorcycle wheels and frame
    cv2.circle(img, (280, 390), 30, (0, 0, 0), -1) # Front wheel
    cv2.circle(img, (280, 390), 15, (128, 128, 128), -1)
    cv2.circle(img, (360, 390), 30, (0, 0, 0), -1) # Back wheel
    cv2.circle(img, (360, 390), 15, (128, 128, 128), -1)
    
    # Frame body
    cv2.line(img, (280, 390), (320, 330), (0, 0, 255), 8) # Front fork
    cv2.line(img, (360, 390), (320, 330), (0, 0, 255), 8) # Rear frame
    cv2.rectangle(img, (295, 325), (345, 355), (0, 0, 180), -1) # Fuel tank
    
    # 2. Rider body (person)
    cv2.rectangle(img, (310, 230), (340, 325), (50, 50, 50), -1) # Torso (dark jacket)
    cv2.line(img, (310, 230), (290, 290), (50, 50, 50), 6) # Arm reaching handlebar
    cv2.circle(img, (285, 290), 8, (200, 150, 120), -1) # Hand
    
    # 3. Rider head (no-helmet: just skin colored circle + black hair)
    cv2.circle(img, (325, 200), 20, (200, 150, 120), -1) # Face/head
    # Draw hair
    cv2.ellipse(img, (325, 192), (18, 12), 0, 180, 360, (0, 0, 0), -1)
    
    # 4. License Plate at the rear of the motorcycle
    # We will embed a tiny version of our license plate image at (350, 360) to (390, 380)
    # Let's draw a white plate rectangle and write MH12DE5678 in small font
    plate_x, plate_y = 350, 365
    plate_w, plate_h = 75, 22
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (255, 255, 255), -1)
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (0, 0, 0), 1)
    cv2.putText(img, "MH12DE5678", (plate_x + 3, plate_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imwrite(filename, img)
    print(f"[Setup] Created synthetic traffic scene: {filename}")

def download_sample_video():
    """Downloads a sample traffic video from the web if it doesn't exist."""
    video_url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
    dest_path = "test_assets/traffic_video_sample.mp4"
    
    if os.path.exists(dest_path):
        print(f"[Setup] Sample video already exists at {dest_path}")
        return
        
    print("[Setup] Downloading sample traffic video from Intel IoT Devkit...")
    try:
        import urllib.request
        req = urllib.request.Request(
            video_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[Setup] Sample video downloaded successfully to {dest_path}")
    except Exception as e:
        print(f"[Setup] Failed to download sample video: {e}")

if __name__ == "__main__":
    os.makedirs("test_assets", exist_ok=True)
    download_yolo_weights()
    
    # Create different types of plates to demonstrate preprocessing benefits
    create_synthetic_plate("MH12AB1234", "test_assets/plate_clean.png", blur=False, skew=False)
    create_synthetic_plate("DL3CAY1111", "test_assets/plate_blurry.png", blur=True, skew=False)
    create_synthetic_plate("KA03MG9999", "test_assets/plate_skewed.png", blur=False, skew=True)
    create_synthetic_plate("HR26BP0007", "test_assets/plate_noisy.png", blur=False, skew=False, noise=True)
    
    # Create sample traffic image
    create_synthetic_traffic_scene("test_assets/traffic_sample.png")
    
    # Download sample traffic video for video processing tab
    download_sample_video()
    
    print("[Setup] Asset configuration completed successfully!")
