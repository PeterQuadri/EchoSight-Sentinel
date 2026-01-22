from PIL import Image
import os

def crop_to_16_9_top_keep(image_path, output_path):
    with Image.open(image_path) as img:
        width, height = img.size
        new_height = int(width * 9 / 16)
        
        # Take the TOP of the source image
        left = 0
        top = 0
        right = width
        bottom = new_height
            
        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(output_path)
        print(f"✅ Image cropped (top kept) and saved to {output_path}")

if __name__ == "__main__":
    src = r"C:\Users\Peter\.gemini\antigravity\brain\3dfafe59-13f0-4c61-8afb-ae78bce60668\echosight_sentinel_audio_visual_v6_1767567724475.png"
    dst = r"C:\Users\Peter\.gemini\antigravity\brain\3dfafe59-13f0-4c61-8afb-ae78bce60668\echosight_sentinel_final_audio_visual_16_9.png"
    crop_to_16_9_top_keep(src, dst)
