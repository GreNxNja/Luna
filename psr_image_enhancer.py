import cv2
import numpy as np
from skimage import exposure

def preprocess_image(image):
    # Apply mild initial noise reduction
    denoised = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21, templateWindowSize=7)
    return denoised

def enhance_contrast(image):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    return enhanced

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def sharpen_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def enhance_psr_image(image_path, gamma=1.2, sharpen_strength=0.5):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess
    preprocessed = preprocess_image(image)
    
    # Enhance contrast
    contrast_enhanced = enhance_contrast(preprocessed)
    
    # Adjust gamma
    gamma_corrected = adjust_gamma(contrast_enhanced, gamma=gamma)
    
    # Sharpen
    sharpened = sharpen_image(gamma_corrected)
    
    # Blend sharpened image with gamma corrected image
    final_enhanced = cv2.addWeighted(gamma_corrected, 1 - sharpen_strength, sharpened, sharpen_strength, 0)
    
    # Final contrast enhancement
    final_enhanced = enhance_contrast(final_enhanced)
    
    return final_enhanced

# Example usage
input_image_path = r'assests\psr_image.png'
enhanced_image = enhance_psr_image(input_image_path, gamma=1.2, sharpen_strength=0.5)

# Save the result
cv2.imwrite('enhanced_psr_image.png', enhanced_image)