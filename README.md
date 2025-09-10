# XylemVision - Django App

This repository contains a **Django** application for performing root structural analysis. The app is set up with **Gunicorn** for production deployment, and it uses weights for **SAM (Segment Anything Model)** and **YOLO (You Only Look Once)**.

## Features

- Root structural analysis using SAM and YOLO.
- Django-based web application for serving the analysis.
- Pre-trained model weights included for SAM and YOLO.

## Requirements

- **Python**: 3.11
- **Django**: >=4.0
- **Gunicorn**: For production deployment
- **PostgreSQL**: (Optional if using PostgreSQL as the database)

## Getting Started

These instructions will help you set up and run the project locally.

### Prerequisites

Ensure you have **Docker** and **Docker Compose** installed on your machine.

- Install Docker: [Docker Installation Guide](https://docs.docker.com/get-docker/)
- Install Docker Compose: [Docker Compose Installation Guide](https://docs.docker.com/compose/install/)

### Dockerize the Application

1. Clone the repository:

    ```bash
    git clone https://github.com/Auburn-Smart-Systems-Lab/root-structural-analysis-client-app.git
    cd root-structural-analysis-client-app
    ```

2. Build the Docker image:

    ```bash
    docker build -t root-structural-analysis .
    ```

3. Run the Docker container:

    ```bash
    docker run --gpus '"device=0"' -p 8000:8000 root-structural-analysis
    ```

    This will start the application, and it will be accessible at `http://localhost:8000`.


### Docker Configuration

The Dockerfile is set up to:

- Use **Python 3.11** and install dependencies listed in `requirements.txt`.
- Automatically download the required model weights (`sam_vit_l_0b3195.pth` and `best.pt`) from Google Drive.
- Use **Gunicorn** to serve the application.

### Configuration

1. The app is configured to run on port `8000` by default.
2. The static files will be stored in the `staticfiles` directory, and the `STATIC_ROOT` setting must be properly configured in the Django `settings.py`.

### Common Issues

- **Missing model weights**: If the download of model weights fails, verify the Google Drive links and permissions.
- **Static files error**: If you're missing static files during `collectstatic`, ensure `STATIC_ROOT` is configured correctly in `settings.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
