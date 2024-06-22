# #WSL2

import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from CurrencyConverter import CurrencyConverter
import re

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press Enter to capture an image or type 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display instructions (in console)
    user_input = input("Press Enter to capture the image, or type 'q' to quit: ").strip().lower()
    
    if user_input == 'q':
        cap.release()
        exit()
    elif user_input == '':
        break

# Release the webcam
cap.release()

# Convert the image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply thresholding
#_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Optionally apply some noise reduction
thresh = cv2.medianBlur(thresh, 3)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify the language(s) you want to recognize

# Perform OCR using EasyOCR
results = reader.readtext(thresh) #

# Display the text detected by EasyOCR
print("Detected text:")
for (bbox, text, prob) in results: 
    print(f'{text} (Confidence: {prob:.2f})')

# Save the preprocessed image (optional)
cv2.imwrite('preprocessed_image.jpg', thresh)


# --- to convert detected values to desired currency exchange --

def convert_string_to_float(text: str):
    pattern = re.compile(r'[^0-9.,]')
    text = pattern.sub('', text)

    text = text.replace(",", ".")

    amount = float(text)

    return amount

amount = convert_string_to_float(text)
currency1 = 'eur'
currency2 = 'jpy'

cc = CurrencyConverter(currency1, currency2)
currency = cc.amount_in_foreign_currency(amount)

print(f"Detected: {amount} {currency1.upper()}")
print(f"Converted: {currency:.2f} {currency2.upper()}")


# Display only the preprocessed image (thresh) using matplotlib
plt.figure(figsize=(8, 6))

plt.imshow(thresh, cmap='gray')
plt.title('Preprocessed Image')
plt.axis('off')

# Add detected and converted currency information as separate strings below the image
plt.text(60, thresh.shape[0] + 50, f"Detected: {amount} {currency1.upper()}",
            horizontalalignment='left', verticalalignment='center', fontsize=36, color='black')
plt.text(60, thresh.shape[0] + 120, f"Converted: {currency:.2f} {currency2.upper()}",
            horizontalalignment='left', verticalalignment='center', fontsize=36, color='black')

plt.tight_layout()
plt.show()




