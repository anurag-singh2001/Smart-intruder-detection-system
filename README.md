# Smart Intruder Detection System

## Overview

The Smart Intruder Detection System is a real-time security solution designed to identify and track intruders in surveillance video streams. Leveraging computer vision techniques with OpenCV and instant alert messaging through Twilio, this project enhances security monitoring capabilities.

## Features

- **Background Subtraction:** Utilizes OpenCV's background subtraction to isolate moving objects from the background in video streams.

- **Contour Detection:** Applies contour detection to refine object recognition, distinguishing potential intruders from other movements.

- **Twilio Integration:** Sends instant alert messages via Twilio when an intruder is detected, ensuring a prompt response to security breaches.

- **Patience Mechanism:** Implements a patience timer to minimize false alarms, allowing time for sustained detections before triggering alerts.

- **Dynamic FPS Display:** Displays real-time Frames Per Second (FPS) information on the video feed for monitoring system performance.

## Prerequisites

- Python 3.x
- OpenCV
- Twilio Account SID and Auth Token
- Camera or video stream source (e.g., IP camera, webcam)

## Setup

1. Install the required dependencies:

    ```bash
    pip install opencv-python twilio
    ```

2. Configure Twilio credentials in `credentials.txt`:

    ```text
    {
      "account_sid": "YOUR_TWILIO_ACCOUNT_SID",
      "auth_token": "YOUR_TWILIO_AUTH_TOKEN",
      "twilio_phone_number": "YOUR_TWILIO_PHONE_NUMBER",
      "recipient_phone_number": "RECIPIENT_PHONE_NUMBER"
    }
    ```

3. Run the main script: Smart Intruder Detection System.ipynb
    

## Usage

Adjust parameters like detection thresholds, patience time, and camera sources in the script to optimize performance for your environment.


