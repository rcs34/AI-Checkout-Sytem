# 🛒 AI Smart Checkout System

An intelligent checkout system that uses computer vision (YOLO) to automatically detect and identify products from images, streamlining the shopping experience with AI-powered product recognition.

## 📋 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Configuration](#configuration)
- [Supported Products](#supported-products)
- [License](#license)

## ✨ Features

- **AI-Powered Product Detection**: Automatically detects products from uploaded images using a custom-trained YOLO model
- **Real-time Object Detection**: Visual feedback with bounding boxes and confidence scores
- **Smart Product Suggestions**: Suggests similar products when detection confidence is uncertain
- **Shopping Cart Management**: Add multiple products with quantity tracking
- **Price Calculation**: Automatic price calculation based on product detection
- **Payment Processing**: Simulated payment system with multiple payment methods (Cash, UPI, Card)
- **Receipt Generation**: Digital receipt after successful checkout
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit**: Web application framework
- **Ultralytics YOLO**: Object detection model
- **OpenCV (cv2)**: Image processing and annotation
- **PIL/Pillow**: Image handling
- **NumPy**: Numerical operations

## 📁 Project Structure

```
AI-Checkout-Sytem/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── prices.json                     # Product pricing database
├── dataset/                        # Training dataset
│   ├── images/
│   │   ├── train/                  # Training images
│   │   └── val/                    # Validation images
│   ├── labels/
│   │   ├── train/                  # Training annotations
│   │   └── val/                    # Validation annotations
│   ├── data.yaml                   # YOLO dataset configuration
│   ├── classes.txt                 # Product class names
│   └── notes.json                  # Dataset metadata
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt         # Trained YOLO model weights
├── misc_scripts/                   # Utility scripts
│   ├── split_dataset.py           # Dataset train/val split utility
│   └── fix_annotations.py          # Annotation fixing utility
├── test_dataset/                   # Test images
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd AI-Checkout-Sytem
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install all necessary dependencies:
   - `streamlit` - Web application framework
   - `ultralytics` - YOLO object detection model
   - `opencv-python` - Image processing
   - `pillow` - Image handling
   - `numpy` - Numerical operations

3. **Ensure model weights are available**:
   - The trained model should be located at: `runs/detect/train/weights/best.pt`
   - If you need to train the model, see the [Model Training](#model-training) section

## 💻 Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - The app will automatically open in your default web browser
   - Default URL: `http://localhost:8501`

### Using the Checkout System

1. **Cart Stage**:
   - Upload an image containing a single product
   - The system will detect and identify the product
   - Review the detection result and select the correct product if needed
   - Click "Add to Cart" to add the product
   - Repeat for additional products
   - Click "Verify & Checkout" when done

2. **Payment Stage**:
   - Review the order summary
   - Select a payment method (Cash, UPI, or Card)
   - Click "Pay Now" to complete the transaction
   - View the generated receipt

### Important Notes

- **Single Product Detection**: The system is designed to detect one product per image. If multiple products are detected, you'll be prompted to upload an image with only one product.
- **Product Confirmation**: You can manually select or search for products if the automatic detection is incorrect.
- **Similar Products**: The system suggests similar products for common misclassifications to improve accuracy.

## 🎯 Model Training

### Training Your Own Model

1. **Prepare your dataset**:
   - Organize images in `dataset/images/train/` and `dataset/images/val/`
   - Create corresponding label files in `dataset/labels/train/` and `dataset/labels/val/`
   - Ensure `dataset/data.yaml` is properly configured

2. **Train the model**:
   ```python
   from ultralytics import YOLO
   
   # Load a pre-trained YOLO model
   model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.
   
   # Train the model
   model.train(
       data='dataset/data.yaml',
       epochs=100,
       imgsz=640,
       batch=16
   )
   ```

3. **Model weights**:
   - After training, the best model will be saved at `runs/detect/train/weights/best.pt`
   - Update `MODEL_PATH` in `app.py` if your model is in a different location

### Dataset Split Utility

Use the provided script to split your dataset:
```bash
python misc_scripts/split_dataset.py
```

**Note**: Update the `BASE_DIR` path in the script to match your dataset location.

## ⚙️ Configuration

### Product Prices

Edit `prices.json` to add or modify product prices:
```json
{
    "Product Name": price_in_rupees,
    "Example Product": 100
}
```

### Detection Settings

In `app.py`, you can adjust:
- `CONF_THRESHOLD`: Confidence threshold for detections (default: 0.5)
- `TOP_K`: Number of top predictions to consider (default: 3)

### Similar Products

Add similar product mappings in `app.py` under `SIMILAR_PRODUCTS` to improve user experience when detections are uncertain.

## 🛍️ Supported Products

The system currently supports the following products:

1. Agnesi - Chifferi Rigati
2. Amul - Pure Milk Cheese Slices
3. Betty Crocker - Complete Pancake Mix Classic
4. Blue Bird - Baking Powder
5. Blue Bird - Baking Soda
6. Ching's Secret - Dark Soy Sauce
7. Home One - Paper Napkin
8. India Gate - Pure Basmati Rice Super
9. Milky Mist - Fresh Cream
10. Nutty Gritties - Mix Berries
11. Patanjali - Atta Noodles
12. Patanjali - Raisin (Black)

To add more products:
1. Add product images to the training dataset
2. Annotate the images with product labels
3. Retrain the model
4. Add the product and price to `prices.json`

## 🔧 Troubleshooting

### Model Not Found Error
- Ensure the model weights file exists at `runs/detect/train/weights/best.pt`
- If training a new model, complete the training process first

### No Products Detected
- Check that the uploaded image contains a clear view of the product
- Ensure the product is one of the supported classes
- Try adjusting the `CONF_THRESHOLD` in `app.py`

### Multiple Products Detected
- The system requires one product per image
- Crop or retake the image to show only one product

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**rcs34**

## 🙏 Acknowledgments

- Ultralytics for the YOLO framework
- Streamlit for the web framework
- OpenCV community for image processing tools

---

**Note**: This is a demonstration system. For production use, consider adding:
- Database integration for product management
- User authentication
- Real payment gateway integration
- Receipt storage and history
- Analytics and reporting features
