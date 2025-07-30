# MLOps Linear Regression\n\nThis project implements a Linear Regression pipeline on the California Housing dataset with MLOps.

This project demonstrates a complete MLOps pipeline using **Linear Regression** on the **California Housing Dataset**.
It includes:
- Data loading & preprocessing
- Model training
- Evaluation (R², MSE)
- Model saving
- Model quantization
- Unit testing with pytest
- CI/CD with GitHub Actions
- Dockerization for deployment
- Source: `sklearn.datasets.fetch_california_housing`
- Features: 8 numerical attributes describing California districts
**Model Comparison Table**
Metric	   Original Model (float32)	  Quantized Model (float16)
Model Size	    ~X MB	                  ~Y MB
R² Score	      0.575                  	~0.575
MSE            	0.556                  	~0.556
InferenceSpeed	Standard	              Faster
