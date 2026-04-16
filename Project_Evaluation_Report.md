# Project Report: Plant Disease Detection System via Vision Transformer

## 1. Project Overview
The "Plant Disease Detection System" is an end-to-end, full-stack AI application designed to provide farmers and gardeners with instantaneous, highly accurate diagnoses of crop diseases. By leveraging advanced Machine Learning techniques—specifically a custom Vision Transformer (ViT)—the system identifies up to 88 distinct classes of plant diseases across more than 15 unique crop types directly from user-uploaded images.

To ensure production-level scalability and reliability, the application utilizes a comprehensive DevOps lifecycle, featuring continuous deployment pipelines, containerized backend services, and native HTTPS security hosted entirely on Amazon Web Services (AWS).

---

## 2. Technology Stack
The project adopts a modern, decoupled client-server architecture:

*   **Frontend User Interface**: React.js, TypeScript, Vite, TailwindCSS
*   **Backend API Service**: Python, Flask, Flask-CORS, Pillow
*   **Machine Learning Engine**: PyTorch, Torchvision, TIMM (PyTorch Image Models)
*   **Infrastructure & DevOps**: Docker, Docker Compose, GitHub Actions (CI/CD)
*   **Cloud Hosting**: AWS EC2 (Backend), AWS Amplify (Frontend)
*   **Security & Routing**: Nginx Reverse Proxy, Let's Encrypt SSL (Certbot)

---

## 3. Machine Learning Architecture
The core of the detection system is powered by a **Vision Transformer (ViT-Base Patch16)** framework. Unlike traditional Convolutional Neural Networks (CNNs), the Vision Transformer splits input images into fixed-size patches and computes self-attention across the sequence, allowing the model to understand complex global relationships and intricate disease patterns across the entire leaf.

**Key ML Implementations:**
*   **Custom Head**: The baseline ViT model was heavily modified to include a custom classification head mapping to exactly 88 biological classes.
*   **Resource Optimization**: The model logic (`model.load_state_dict(strict=False)`) was fortified to catch shape-mismatches and drop non-matching tensors automatically.
*   **CPU Optimization Engine**: To reduce cloud hosting costs, the Dockerized PyTorch engine was built using a CPU-only index (`download.pytorch.org/whl/cpu`), stripping over 3 Gigabytes of unnecessary NVIDIA CUDA libraries. This allows the heavy ViT model to infer extremely fast on a standard low-cost AWS EC2 instance.

---

## 4. Advanced "Smart Gating" Heuristics
A major challenge in public-facing AI image systems is the risk of users uploading irrelevant images (e.g., selfies, pets, documents) that force the AI to return false-positive biological predictions.

To solve this, a completely custom **Heuristic Gating Algorithm** was engineered into the pre-processing pipeline:
*   Before the image touches the neural network, the backend converts it into the **HSV (Hue, Saturation, Value)** color space.
*   The algorithm scans the pixels for distinct non-biological signatures—specifically targeting human skin-tone thresholds and non-organic saturations (heavily analyzing the center pixels for faces).
*   If an image registers mathematically as a "Selfie" or lacks plant-like green/brown hues, the pipeline halts and forces a `0.0% Confidence` score with the error: *"Not a leaf image"*, completely safeguarding the model's integrity.

---

## 5. Security & DevOps Deployment Pipeline
The system was removed from local execution and upgraded into a fully automated, production-grade DevOps environment mirroring enterprise standards.

### A. Containerization & Automation
*   **Docker Integration**: The Python ML backend is completely containerized inside an isolated `python:3.10-slim` container, defined by `Dockerfile` and orchestrated via `docker-compose.yml`. This guarantees the codebase runs identically regardless of the underlying hardware.
*   **GitHub Actions CI/CD**: A `.github/workflows/ci.yml` pipeline triggers automatically on every code push, statically analyzing the Docker build integrity to prevent broken code from ever reaching the live server.

### B. AWS Infrastructure Edge
*   **Frontend (AWS Amplify)**: The React UI is hosted on a globally distributed Content Delivery Network (CDN). It utilizes AWS continuous deployment to automatically rebuild the interface whenever UI code receives an update.
*   **Backend (AWS EC2)**: The AI container is hosted on a scalable AWS EC2 compute instance. 

### C. HTTPS & Reverse Proxy Security (saket.tech)
To achieve secure cross-origin communication between the UI and Backend, the raw EC2 instance was heavily secured using a custom domain (`api.saket.tech`).
*   **Nginx**: Installed as a Reverse Proxy on the EC2 machine to safely intercept incoming internet traffic on Port 443 and stream it natively to the internal Docker container on Port 5000.
*   **Certbot**: Linked with the Electronic Frontier Foundation to automatically provision and rotate an official Let's Encrypt 2048-bit RSA **SSL Certificate**.
*   **Payload Protection**: Nginx was explicitly configured (`client_max_body_size 20M`) to securely handle dynamically sized high-resolution phone images while mitigating DDoS flooding attacks.

---

## 6. Conclusion
By fusing state-of-the-art transformer neural networks with rigorous backend heuristics and an enterprise-tier AWS architectural deployment, the Plant Disease Detection system has evolved from a conceptual script into a robust, highly-available cloud application. It represents a concrete demonstration of end-to-end Machine Learning Operations (MLOps), proving capable of securely diagnosing agricultural threats in real-time across the globe.
