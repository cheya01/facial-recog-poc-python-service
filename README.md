# Facial Recognition Python Service

A Flask-based facial verification service using DeepFace for identity verification.

## Features

- Face verification using DeepFace library
- Image preprocessing for improved accuracy
- RESTful API endpoint
- Support for multiple image formats
- Logging and error handling

## Local Development

### Prerequisites

- Python 3.12
- pip

### Installation

1. Create a virtual environment:
```bash
python -m venv my-venv
source my-venv/bin/activate  # On Windows: my-venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the service:
```bash
python face_verifire.py
```

The service will start on `http://localhost:5001`

## API Usage

### Verify Faces

**Endpoint:** `POST /verify`

**Request:** Form-data with two image files
- `img1`: First face image
- `img2`: Second face image

**Response:**
```json
{
  "verified": true,
  "distance": 0.25,
  "threshold": 0.4,
  "model": "VGG-Face",
  "similarity_metric": "cosine",
  "request_id": "abc123"
}
```

## Deployment on Render

### Option 1: Using Render Dashboard

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" and select "Web Service"
3. Connect your GitHub repository: `https://github.com/cheya01/facial-recog-poc-python-service`
4. Configure the service:
   - **Name:** facial-recog-python-service
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn face_verifire:app`
   - **Plan:** Free (or select your preferred plan)
5. Click "Create Web Service"

### Option 2: Using render.yaml (Infrastructure as Code)

1. Push your code to GitHub (already done)
2. In Render Dashboard, click "New +" and select "Blueprint"
3. Connect your repository
4. Render will automatically detect the `render.yaml` file and configure the service

### Important Notes for Render Deployment

- The free tier on Render may have limited resources. Face recognition is CPU-intensive.
- Consider upgrading to a paid plan for better performance
- The service may take a few minutes to start on first deployment while installing dependencies
- Render automatically sets the `PORT` environment variable

## Environment Variables

- `PORT`: The port the service runs on (automatically set by Render)

## Technologies Used

- Flask - Web framework
- DeepFace - Face recognition library
- OpenCV - Image processing
- Gunicorn - Production WSGI server
- TensorFlow - Deep learning backend

## License

MIT
