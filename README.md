# Number Plate Recognition 🚗🔍

This is a traffic surveillance module that detects vehicles and recognizes license plates from video using **YOLOv8** for object detection and **SORT** for object tracking.

## 📁 Project Structure

```
number_plate_recognition/
├── data/                  # Input/output data (videos, CSVs)
├── models/                # Pretrained YOLOv8 models
├── sort/                  # SORT tracking algorithm
├── src/                   # Main source code
│   ├── main.py            # Entry point
│   ├── driver.py          # Main driver logic
│   ├── splitframe.py      # Extract frames from video
│   ├── visualizetation.py # Visualization helpers
│   ├── utilts.py          # Utility functions
│   └── ...
├── test/                  # Testing and experiments
├── requirements.txt       # Python dependencies
├── LICENSE
└── README.md              # This file
```

## 🚀 Features

- Vehicle detection using fine-tuned YOLOv8 models
- License plate recognition using OCR (EasyOCR)
- Multi-object tracking with SORT
- Frame-by-frame processing and visualization
- Export results to CSV
- Optional: Frame interpolation and data augmentation

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/PhucNguyenThanh04/Number_plate_Vehicles.git
cd number_plate_recognition
```


2Install dependencies:

```bash
pip install -r requirements.txt
```

## 📦 Pretrained Models

Put the following pretrained models inside the `models/` directory:

- `yolov8n.pt` – Base model
- `yolov8n_fine_tune_number_plate.pt` – Fine-tuned model for license plate detection
- `finetune_v8_vehicles.pt` – Fine-tuned model for vehicle detection

## 🎬 Usage

Run the main program to detect and track vehicles + plates in a video:

```bash
python .\visualizetation.py --input_video path_video.mp4 --csv filename.csv --csv_full filename.csv
```

> Output will be visualized in real time and saved as `out_full.mp4`.

You can also split frames manually:

```bash
python src/splitframe.py
```

Or run OCR on a CSV of detections:

```bash
python src/driver.py
```

## 📊 Output

- **CSV**: All recognized license plates and vehicle IDs with timestamps (`results.csv`)
- **MP4**: Video with bounding boxes and plate numbers overlaid

## 🧪 Testing

Simple test script (e.g. FPS, functionality):

```bash
python test/test.py
```

## 🧾 License

This project is licensed under the [MIT License](LICENSE).

## ✍️ Author

Developed by [Your Name] as part of a traffic surveillance camera system project.
