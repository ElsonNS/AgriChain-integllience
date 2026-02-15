# Design Document: AgriChain Intelligence

## Overview

AgriChain Intelligence is a cloud-native, AI-powered platform built on AWS infrastructure to provide rural producer groups with market intelligence and collective bargaining tools. The system leverages serverless architecture for scalability, Amazon SageMaker for ML model training and inference, and multi-channel delivery (web dashboard and SMS) to accommodate varying levels of digital connectivity in rural areas.

The platform ingests historical mandi price data, aggregates farmer supply information, runs AI models for price forecasting and fair price estimation, simulates collective selling scenarios, and delivers actionable insights to FPO leaders through an intuitive dashboard and SMS alerts to farmers.

## System Architecture Overview

### High-Level Architecture

The AgriChain Intelligence system follows a cloud-native, serverless architecture pattern built entirely on AWS services. The architecture is designed for:

- **Scalability**: Serverless components auto-scale based on demand
- **Cost-efficiency**: Pay-per-use model suitable for hackathon and pilot deployments
- **Low-latency**: Edge caching and regional deployment for fast response times
- **Resilience**: Managed services with built-in redundancy and fault tolerance

### Architecture Layers

1. **Presentation Layer**: Web dashboard (React/Vue.js) and SMS interface
2. **API Layer**: Amazon API Gateway with Lambda functions
3. **Business Logic Layer**: AWS Lambda functions for orchestration
4. **ML/AI Layer**: Amazon SageMaker for model training and inference
5. **Data Layer**: S3 for raw data, DynamoDB for operational data, RDS for relational data
6. **Integration Layer**: AWS Glue for ETL, SNS for notifications
7. **Monitoring Layer**: CloudWatch for logging and metrics

### Data Flow Overview

```
Farmer Input (Web/SMS) 
  → API Gateway 
  → Lambda (Input Validation) 
  → DynamoDB (Store Supply Data)
  → Lambda (Aggregation Logic)
  → SageMaker (Price Forecast + Fair Price Estimation)
  → Lambda (Scenario Simulation)
  → DynamoDB (Store Results)
  → Dashboard (Display Insights) + SNS (Send SMS Alerts)
```


## Component Architecture

### 1. Farmer Input Layer

**Purpose**: Capture farmer supply data and buyer offers through multiple channels

**Components**:
- **Web Interface**: React-based responsive web application hosted on S3 + CloudFront
  - Forms for FPO leaders to input farmer supply data (crop type, quantity, location)
  - Buyer offer entry form for fair price validation
  - Mobile-optimized for smartphone access
  
- **SMS Interface**: Bidirectional SMS gateway using Amazon SNS and Pinpoint
  - Inbound: Farmers send structured SMS (e.g., "SUPPLY WHEAT 50 QUINTAL")
  - Outbound: System sends alerts and recommendations
  - SMS parsing Lambda function extracts structured data

**Technology Stack**:
- Frontend: React.js with Material-UI, hosted on S3 + CloudFront
- SMS Gateway: Amazon SNS for outbound, Amazon Pinpoint for inbound
- API Integration: REST API via API Gateway

### 2. Data Ingestion Layer

**Purpose**: Validate, transform, and route incoming data to appropriate storage

**Components**:
- **API Gateway**: RESTful API endpoints for all client interactions
  - `/api/supply` - POST farmer supply data
  - `/api/offers` - POST buyer offers for validation
  - `/api/forecast` - GET price forecasts
  - `/api/simulate` - POST simulation requests
  
- **Input Validation Lambda**: Validates data format, range checks, authentication
  - Validates crop types against supported list
  - Ensures quantities are positive numbers
  - Checks user authentication tokens (JWT)
  
- **SMS Parser Lambda**: Parses inbound SMS into structured JSON
  - Regex-based parsing for predefined SMS formats
  - Error handling for malformed messages
  - Sends confirmation SMS back to farmer

**Technology Stack**:
- Amazon API Gateway (REST API)
- AWS Lambda (Node.js or Python)
- Amazon Cognito for authentication

### 3. Data Storage Layer

**Purpose**: Persist historical data, operational data, and ML model artifacts

**Components**:

**Amazon S3 Buckets**:
- `agrichain-mandi-data`: Historical mandi price CSV files from Agmarknet
- `agrichain-model-artifacts`: Trained ML models, scalers, encoders
- `agrichain-exports`: CSV exports for FPO leaders
- `agrichain-logs`: Application and audit logs

**Amazon DynamoDB Tables**:
- `Farmers`: Farmer profiles (FarmerID, Name, FPO, Location, Phone)
- `SupplyData`: Current supply submissions (FarmerID, CropType, Quantity, Date)
- `BuyerOffers`: Buyer offers for validation (OfferID, CropType, Price, Timestamp)
- `Simulations`: Cached simulation results (SimulationID, Scenario, Results, Timestamp)
- `Alerts`: SMS alert history (AlertID, RecipientPhone, Message, Status, Timestamp)

**Amazon RDS (PostgreSQL)** (Optional for complex queries):
- Relational storage for aggregated analytics
- Time-series price data with indexing for fast queries
- Used if DynamoDB query patterns become complex

**Technology Stack**:
- Amazon S3 (object storage)
- Amazon DynamoDB (NoSQL, primary operational database)
- Amazon RDS PostgreSQL (optional, for analytics)

### 4. Data Processing Layer

**Purpose**: ETL pipelines for mandi data ingestion and preprocessing

**Components**:

**AWS Glue Jobs**:
- **Mandi Data ETL**: Daily job to fetch latest Agmarknet data
  - Downloads CSV files from Agmarknet API or S3 drop location
  - Cleans data (handle missing values, outliers)
  - Transforms to standardized schema
  - Loads into S3 and DynamoDB/RDS
  
- **Feature Engineering**: Prepares features for ML models
  - Calculates rolling averages, seasonality indicators
  - Encodes categorical variables (crop type, location)
  - Generates training datasets for SageMaker

**Lambda Functions**:
- **Supply Aggregation Lambda**: Aggregates farmer supply by FPO and crop type
  - Queries DynamoDB `SupplyData` table
  - Groups by FPO and CropType
  - Caches results in DynamoDB `Simulations` table
  
- **Logistics Calculator Lambda**: Estimates transportation costs
  - Distance calculation using location coordinates
  - Applies cost-per-km formula with economies of scale
  - Returns pooled vs individual cost comparison

**Technology Stack**:
- AWS Glue (PySpark ETL jobs)
- AWS Lambda (Python 3.9+)
- Amazon EventBridge for scheduled triggers

### 5. AI/ML Layer

**Purpose**: Train and deploy ML models for price forecasting and fair price estimation

**Components**:

**Amazon SageMaker Training Jobs**:
- **Time-Series Forecasting Model**:
  - Algorithm: DeepAR (SageMaker built-in) or custom LSTM
  - Training data: 24 months of daily mandi prices from S3
  - Hyperparameters: Context length (30 days), prediction length (30 days)
  - Output: Trained model artifact stored in S3
  
- **Fair Price Regression Model**:
  - Algorithm: XGBoost (SageMaker built-in) or Random Forest
  - Features: Historical prices, seasonality, crop type, location, supply volume
  - Training data: Historical mandi transactions with fair price labels
  - Output: Trained model artifact stored in S3

**Amazon SageMaker Endpoints**:
- **Forecast Endpoint**: Real-time inference for price predictions
  - Input: Crop type, location, current date
  - Output: 30-day price forecast with confidence intervals
  - Instance type: ml.m5.large (cost-optimized for hackathon)
  
- **Fair Price Endpoint**: Real-time inference for price validation
  - Input: Crop type, location, current date, supply quantity
  - Output: Estimated fair price and confidence score (0-100)
  - Instance type: ml.m5.large

**Model Retraining Pipeline**:
- Weekly EventBridge trigger initiates SageMaker training job
- Automated model evaluation compares new model vs current production model
- If new model MAPE < current model MAPE, deploy to endpoint
- Rollback capability if inference errors spike

**Technology Stack**:
- Amazon SageMaker (training and inference)
- SageMaker built-in algorithms (DeepAR, XGBoost)
- Custom algorithms (LSTM, Random Forest) via Docker containers
- S3 for model artifacts and training data

### 6. Decision Simulation Engine

**Purpose**: Run scenario simulations comparing individual vs collective selling

**Components**:

**Simulation Lambda Function**:
- **Input**: Aggregated supply quantity, current price, forecasted prices, commission rates
- **Logic**:
  - **Scenario 1 (Individual Selling)**: 
    - Revenue = Quantity × Current_Price × (1 - Individual_Commission_Rate)
    - Individual_Commission_Rate = 10% (average)
  
  - **Scenario 2 (Immediate Group Selling)**:
    - Revenue = Quantity × Current_Price × (1 - Group_Commission_Rate)
    - Group_Commission_Rate = 4% (bulk discount)
  
  - **Scenario 3 (Delayed Group Selling)**:
    - Revenue = Quantity × Forecasted_Price_Peak × (1 - Group_Commission_Rate) - Storage_Cost
    - Forecasted_Price_Peak = Max price in next 30 days from forecast
    - Storage_Cost = Quantity × Storage_Cost_Per_Quintal_Per_Day × Days_Delayed
  
- **Output**: JSON with revenue estimates, percentage improvements, recommended action

**Caching Strategy**:
- Cache simulation results in DynamoDB for 24 hours
- Cache key: Hash of (FPO_ID, Crop_Type, Date)
- Reduces redundant SageMaker inference calls

**Technology Stack**:
- AWS Lambda (Python 3.9+)
- DynamoDB for caching

### 7. Notification Layer

**Purpose**: Deliver alerts and insights via SMS and email

**Components**:

**Amazon SNS Topics**:
- `price-alerts`: Publishes price movement alerts (>10% change)
- `collective-selling-opportunities`: Publishes simulation results showing >20% improvement
- `fair-price-warnings`: Publishes alerts when buyer offers are flagged as unfair

**SMS Delivery Lambda**:
- Subscribes to SNS topics
- Formats messages to 160 characters
- Calls Amazon SNS SMS publish API
- Logs delivery status to DynamoDB `Alerts` table

**Email Delivery** (Optional):
- Amazon SES for email notifications to FPO leaders
- HTML-formatted reports with charts and tables

**Technology Stack**:
- Amazon SNS (pub/sub messaging)
- Amazon SES (email delivery)
- AWS Lambda (message formatting)

### 8. Dashboard Layer

**Purpose**: Web-based UI for FPO leaders to access all features

**Components**:

**Frontend Application**:
- **Technology**: React.js with Material-UI or Ant Design
- **Hosting**: S3 static website + CloudFront CDN
- **Features**:
  - Supply aggregation view (table + charts)
  - Price forecast charts (line graphs with confidence bands)
  - Simulation comparison (bar charts showing revenue scenarios)
  - Fair price validation form (input buyer offer, see score)
  - Logistics cost calculator
  - SMS alert history
  - CSV export buttons

**Backend API**:
- API Gateway + Lambda functions
- JWT authentication via Amazon Cognito
- Role-based access control (FPO Leader vs Farmer roles)

**Optional: Amazon QuickSight**:
- Embedded analytics dashboards
- Pre-built visualizations for price trends, supply aggregation
- Direct connection to DynamoDB or RDS
- Useful for advanced analytics but not required for MVP

**Technology Stack**:
- React.js (frontend)
- Amazon S3 + CloudFront (hosting)
- Amazon API Gateway + Lambda (backend)
- Amazon Cognito (authentication)
- Amazon QuickSight (optional analytics)


## AI Model Design

### Time-Series Forecasting Model

**Objective**: Predict daily mandi prices for the next 30 days

**Approach**: DeepAR (Amazon SageMaker built-in algorithm)

**Why DeepAR**:
- Handles multiple time series simultaneously (different crops, locations)
- Produces probabilistic forecasts (confidence intervals)
- Works well with limited data per time series
- Native SageMaker integration (no custom Docker required)

**Model Architecture**:
- Recurrent neural network with LSTM cells
- Context length: 30 days (uses last 30 days to predict next 30)
- Prediction length: 30 days
- Covariates: Seasonality indicators (month, week), crop type, location

**Training Data Format**:
```json
{
  "start": "2022-01-01 00:00:00",
  "target": [120, 122, 125, ...],  // Daily prices
  "cat": [0, 1],  // Categorical features: [crop_type_encoded, location_encoded]
  "dynamic_feat": [[1, 0, 0, ...], [0, 1, 0, ...]]  // Seasonality one-hot encoding
}
```

**Training Configuration**:
- Instance type: ml.m5.xlarge
- Training time: ~30 minutes for 15 crops × 50 locations × 24 months
- Hyperparameters:
  - `epochs`: 100
  - `context_length`: 30
  - `prediction_length`: 30
  - `num_cells`: 40
  - `likelihood`: "gaussian"

**Inference**:
- Input: Crop type, location, last 30 days of prices
- Output: 30-day forecast with P10, P50, P90 quantiles
- Latency: <2 seconds per inference

**Performance Target**: MAPE < 15%

**Alternative Approach** (if DeepAR underperforms):
- ARIMA or Prophet for simpler time-series modeling
- Separate model per crop type
- Faster training but less sophisticated

### Fair Price Regression Model

**Objective**: Estimate fair market price for a given crop, location, and date

**Approach**: XGBoost Regression (Amazon SageMaker built-in algorithm)

**Why XGBoost**:
- Handles tabular data with mixed feature types
- Robust to outliers and missing values
- Fast training and inference
- Interpretable feature importance

**Features**:
- `crop_type` (categorical, one-hot encoded)
- `location` (categorical, one-hot encoded)
- `date` (decomposed into month, week, day_of_year)
- `supply_quantity` (numerical)
- `rolling_avg_7d` (7-day rolling average of historical prices)
- `rolling_avg_30d` (30-day rolling average)
- `price_volatility` (standard deviation of last 30 days)
- `seasonality_index` (crop-specific seasonal multiplier)

**Target Variable**: Historical mandi price (fair price label)

**Training Data**:
- Historical mandi transactions from Agmarknet (24 months)
- ~100,000 records after preprocessing
- 80/20 train/test split

**Training Configuration**:
- Instance type: ml.m5.xlarge
- Training time: ~10 minutes
- Hyperparameters:
  - `num_round`: 100
  - `max_depth`: 6
  - `eta`: 0.3
  - `objective`: "reg:squarederror"

**Inference**:
- Input: Crop type, location, date, supply quantity
- Output: Estimated fair price (single value)
- Latency: <1 second per inference

**Fair Price Confidence Score Calculation**:
```python
def calculate_confidence_score(buyer_offer, estimated_fair_price):
    ratio = buyer_offer / estimated_fair_price
    if ratio >= 0.95:
        return 100  # Excellent offer
    elif ratio >= 0.90:
        return 80   # Good offer
    elif ratio >= 0.85:
        return 60   # Fair offer
    elif ratio >= 0.80:
        return 40   # Below fair
    else:
        return 20   # Potentially unfair
```

**Performance Target**: R² > 0.75 on test set

### Scenario Simulation Logic

**Objective**: Compare revenue outcomes for three selling strategies

**Approach**: Rule-based calculation (not ML-based)

**Scenario 1: Individual Selling**
```python
def individual_selling(quantity, current_price):
    commission_rate = 0.10  # 10% commission
    revenue = quantity * current_price * (1 - commission_rate)
    return revenue
```

**Scenario 2: Immediate Group Selling**
```python
def immediate_group_selling(quantity, current_price):
    commission_rate = 0.04  # 4% commission (bulk discount)
    revenue = quantity * current_price * (1 - commission_rate)
    return revenue
```

**Scenario 3: Delayed Group Selling**
```python
def delayed_group_selling(quantity, forecasted_prices, storage_cost_per_day):
    commission_rate = 0.04
    
    # Find optimal selling day (max revenue after storage costs)
    max_revenue = 0
    optimal_day = 0
    
    for day, price in enumerate(forecasted_prices):
        storage_cost = quantity * storage_cost_per_day * day
        revenue = quantity * price * (1 - commission_rate) - storage_cost
        if revenue > max_revenue:
            max_revenue = revenue
            optimal_day = day
    
    return max_revenue, optimal_day
```

**Output Format**:
```json
{
  "scenarios": {
    "individual": {
      "revenue": 45000,
      "commission": 5000,
      "net_revenue": 45000
    },
    "immediate_group": {
      "revenue": 48000,
      "commission": 2000,
      "net_revenue": 48000,
      "improvement_pct": 6.7
    },
    "delayed_group": {
      "revenue": 55000,
      "commission": 2200,
      "storage_cost": 800,
      "net_revenue": 52000,
      "improvement_pct": 15.6,
      "optimal_day": 12
    }
  },
  "recommendation": "delayed_group",
  "reasoning": "Waiting 12 days yields 15.6% higher revenue"
}
```

### Fairness Scoring Methodology

**Objective**: Quantify how fair a buyer offer is compared to market price

**Approach**: Comparison-based scoring with thresholds

**Steps**:
1. Get estimated fair price from XGBoost model
2. Calculate ratio: `buyer_offer / estimated_fair_price`
3. Apply scoring thresholds (see Fair Price Confidence Score Calculation above)
4. Add contextual flags:
   - "Excellent" (score 80-100): Offer is at or above market rate
   - "Fair" (score 60-79): Offer is slightly below but acceptable
   - "Caution" (score 40-59): Offer is notably below market
   - "Warning" (score 0-39): Offer may be exploitative

**Explainability**:
- Show estimated fair price alongside buyer offer
- Display percentage difference
- Highlight key factors (e.g., "Current market average is ₹1250/quintal")
- Provide historical price context (7-day and 30-day averages)

### Logistics Cost Calculation Logic

**Objective**: Estimate transportation cost savings from pooled logistics

**Approach**: Distance-based cost model with economies of scale

**Individual Farmer Transportation**:
```python
def individual_logistics_cost(distance_km, quantity_quintals):
    # Small vehicle (pickup truck)
    base_cost = 500  # Fixed cost per trip
    cost_per_km = 15  # ₹15 per km
    loading_cost = 200  # Loading/unloading
    
    total_cost = base_cost + (distance_km * cost_per_km) + loading_cost
    cost_per_quintal = total_cost / quantity_quintals
    return cost_per_quintal
```

**Pooled Logistics**:
```python
def pooled_logistics_cost(distance_km, total_quantity_quintals):
    # Large vehicle (truck)
    truck_capacity = 200  # quintals
    num_trucks = math.ceil(total_quantity_quintals / truck_capacity)
    
    base_cost_per_truck = 2000
    cost_per_km_per_truck = 25
    loading_cost_per_truck = 500
    
    total_cost = num_trucks * (base_cost_per_truck + (distance_km * cost_per_km_per_truck) + loading_cost_per_truck)
    cost_per_quintal = total_cost / total_quantity_quintals
    return cost_per_quintal
```

**Savings Calculation**:
```python
individual_cost = individual_logistics_cost(distance, quantity)
pooled_cost = pooled_logistics_cost(distance, total_quantity)
savings_pct = ((individual_cost - pooled_cost) / individual_cost) * 100
```

**Typical Savings**: 20-35% for distances >50 km and quantities >100 quintals


## Data Flow Design

### End-to-End Data Flow

**Step 1: Farmer Data Submission**

1. FPO leader logs into web dashboard (or farmer sends SMS)
2. Enters supply data: Crop type, quantity, location, harvest date
3. Frontend validates input (non-empty fields, positive quantities)
4. POST request to API Gateway `/api/supply`
5. API Gateway triggers Input Validation Lambda
6. Lambda validates data format and authenticates user (Cognito JWT)
7. Lambda writes record to DynamoDB `SupplyData` table
8. Lambda returns success response to frontend

**Step 2: Supply Aggregation**

1. FPO leader clicks "View Aggregated Supply" on dashboard
2. GET request to API Gateway `/api/supply/aggregate?fpo_id=123`
3. API Gateway triggers Supply Aggregation Lambda
4. Lambda queries DynamoDB `SupplyData` table filtered by FPO_ID
5. Lambda groups records by crop type and sums quantities
6. Lambda caches result in DynamoDB `Simulations` table (TTL: 24 hours)
7. Lambda returns aggregated data to frontend
8. Frontend displays table and bar chart of aggregated supply

**Step 3: Market Data Retrieval**

1. EventBridge scheduled rule triggers daily at 6 AM IST
2. Triggers AWS Glue ETL job "Mandi Data Ingestion"
3. Glue job downloads latest CSV from Agmarknet API or S3 drop location
4. Glue job cleans data (handle nulls, remove outliers >3 std dev)
5. Glue job transforms to standardized schema
6. Glue job writes cleaned data to S3 `agrichain-mandi-data/processed/`
7. Glue job updates DynamoDB `MandiPrices` table with latest prices
8. Glue job triggers SageMaker retraining pipeline (weekly only)

**Step 4: Model Inference (Price Forecast)**

1. FPO leader clicks "View Price Forecast" for a crop
2. GET request to API Gateway `/api/forecast?crop=wheat&location=punjab`
3. API Gateway triggers Forecast Lambda
4. Lambda checks DynamoDB cache for recent forecast (TTL: 24 hours)
5. If cache miss, Lambda calls SageMaker Forecast Endpoint
6. SageMaker endpoint loads DeepAR model from S3
7. SageMaker runs inference using last 30 days of prices from DynamoDB
8. SageMaker returns 30-day forecast with P10, P50, P90 quantiles
9. Lambda caches result in DynamoDB
10. Lambda returns forecast to frontend
11. Frontend displays line chart with confidence bands

**Step 5: Model Inference (Fair Price Estimation)**

1. FPO leader enters buyer offer in "Validate Offer" form
2. POST request to API Gateway `/api/offers/validate`
3. Request body: `{crop: "wheat", location: "punjab", quantity: 500, buyer_offer: 1200}`
4. API Gateway triggers Fair Price Lambda
5. Lambda calls SageMaker Fair Price Endpoint
6. SageMaker endpoint loads XGBoost model from S3
7. SageMaker runs inference using input features + historical price features
8. SageMaker returns estimated fair price (e.g., 1280)
9. Lambda calculates confidence score: `1200/1280 = 0.9375 → Score: 80`
10. Lambda writes offer record to DynamoDB `BuyerOffers` table
11. Lambda returns fair price, score, and explanation to frontend
12. Frontend displays score with color coding (green/yellow/red)

**Step 6: Scenario Simulation**

1. FPO leader clicks "Run Simulation" button
2. POST request to API Gateway `/api/simulate`
3. Request body: `{fpo_id: 123, crop: "wheat", quantity: 5000}`
4. API Gateway triggers Simulation Lambda
5. Lambda retrieves current price from DynamoDB `MandiPrices`
6. Lambda retrieves forecasted prices from DynamoDB cache (or calls SageMaker)
7. Lambda runs three scenario calculations (see Scenario Simulation Logic)
8. Lambda writes simulation result to DynamoDB `Simulations` table
9. Lambda returns scenario comparison JSON to frontend
10. Frontend displays bar chart comparing revenues and improvement percentages

**Step 7: Decision Recommendation**

1. Based on simulation results, Lambda determines recommended action
2. If delayed group selling shows >15% improvement, recommend "Wait and sell collectively"
3. If immediate group selling shows >10% improvement, recommend "Sell now collectively"
4. Otherwise, recommend "Monitor prices, consider individual selling"
5. Lambda includes reasoning in response (e.g., "Prices expected to peak in 12 days")

**Step 8: Alert Generation**

1. EventBridge scheduled rule triggers Price Monitor Lambda every 6 hours
2. Lambda queries DynamoDB `MandiPrices` for latest prices
3. Lambda compares current price vs 24-hour-ago price
4. If price change >10%, Lambda publishes message to SNS topic `price-alerts`
5. SMS Delivery Lambda subscribes to SNS topic
6. Lambda formats message: "WHEAT price up 12% to ₹1300/quintal. Consider selling."
7. Lambda calls SNS SMS publish API for all farmers subscribed to wheat alerts
8. Lambda logs SMS delivery status to DynamoDB `Alerts` table
9. Farmers receive SMS within 5 minutes

**Step 9: Dashboard Refresh**

1. Frontend polls API Gateway `/api/dashboard` every 30 seconds
2. Lambda returns latest aggregated supply, forecasts, simulations, alerts
3. Frontend updates UI components without full page reload
4. User sees real-time updates as new data arrives


## AWS Service Mapping

### Complete Service Architecture

| Component | AWS Service | Purpose | Configuration |
|-----------|-------------|---------|---------------|
| **Frontend Hosting** | S3 + CloudFront | Static website hosting with CDN | S3 bucket with static website enabled, CloudFront distribution for HTTPS and caching |
| **API Layer** | API Gateway | RESTful API endpoints | REST API with Lambda proxy integration, CORS enabled, API key authentication |
| **Authentication** | Cognito | User authentication and authorization | User pool with email/password, JWT tokens, MFA optional |
| **Business Logic** | Lambda | Serverless compute for all backend logic | Python 3.9 runtime, 512 MB memory, 30s timeout, VPC optional |
| **ML Training** | SageMaker Training | Train DeepAR and XGBoost models | ml.m5.xlarge instances, spot instances for cost savings |
| **ML Inference** | SageMaker Endpoints | Real-time model inference | ml.m5.large instances, auto-scaling 1-3 instances |
| **Operational Data** | DynamoDB | NoSQL database for supply, offers, simulations | On-demand billing, TTL enabled for cache expiration, GSI for queries |
| **Relational Data** | RDS PostgreSQL (Optional) | Time-series price data with complex queries | db.t3.micro for hackathon, Multi-AZ for production |
| **Object Storage** | S3 | Raw data, model artifacts, exports, logs | Standard storage class, lifecycle policies for archival |
| **ETL Pipelines** | Glue | Data ingestion and preprocessing | Python shell jobs for lightweight ETL, Spark jobs for heavy processing |
| **Scheduling** | EventBridge | Trigger ETL jobs, model retraining, alerts | Cron expressions for daily/weekly schedules |
| **Notifications** | SNS | Pub/sub for SMS and email alerts | Standard topics, SMS delivery with opt-in/opt-out |
| **SMS Gateway** | Pinpoint (Optional) | Bidirectional SMS for farmer input | SMS channel enabled, long code or short code |
| **Monitoring** | CloudWatch | Logs, metrics, alarms | Log groups for Lambda, custom metrics for model performance |
| **Analytics** | QuickSight (Optional) | Embedded dashboards for FPO leaders | Standard edition, direct query to DynamoDB/RDS |
| **Secrets Management** | Secrets Manager | Store API keys, database credentials | Automatic rotation for RDS passwords |
| **IAM** | IAM | Role-based access control | Least privilege policies for Lambda, SageMaker, Glue |

### Detailed Service Configurations

**Amazon S3 Buckets**:
```
agrichain-frontend/
  ├── index.html
  ├── static/
  │   ├── css/
  │   ├── js/
  │   └── images/

agrichain-mandi-data/
  ├── raw/
  │   └── agmarknet_2022_2024.csv
  ├── processed/
  │   └── cleaned_prices.parquet
  └── features/
      └── training_data.csv

agrichain-model-artifacts/
  ├── deepar/
  │   ├── model.tar.gz
  │   └── metadata.json
  └── xgboost/
      ├── model.tar.gz
      └── feature_scaler.pkl

agrichain-exports/
  └── fpo_123_supply_report_2024_01_15.csv

agrichain-logs/
  └── lambda_logs/
```

**DynamoDB Tables**:

`Farmers` Table:
- Partition Key: `FarmerID` (String)
- Attributes: `Name`, `FPO_ID`, `Location`, `Phone`, `Crops`
- GSI: `FPO_ID-index` for querying all farmers in an FPO

`SupplyData` Table:
- Partition Key: `SupplyID` (String, UUID)
- Sort Key: `Timestamp` (Number, Unix timestamp)
- Attributes: `FarmerID`, `FPO_ID`, `CropType`, `Quantity`, `Location`, `HarvestDate`
- GSI: `FPO_ID-CropType-index` for aggregation queries
- TTL: `ExpirationTime` (auto-delete records older than 90 days)

`BuyerOffers` Table:
- Partition Key: `OfferID` (String, UUID)
- Attributes: `FPO_ID`, `CropType`, `BuyerPrice`, `EstimatedFairPrice`, `ConfidenceScore`, `Timestamp`

`Simulations` Table:
- Partition Key: `SimulationID` (String, hash of FPO_ID + CropType + Date)
- Attributes: `FPO_ID`, `CropType`, `Scenarios`, `Recommendation`, `Timestamp`
- TTL: `ExpirationTime` (24 hours)

**Lambda Functions**:

`InputValidationLambda`:
- Runtime: Python 3.9
- Memory: 256 MB
- Timeout: 10s
- Environment Variables: `DYNAMODB_TABLE_NAME`, `COGNITO_USER_POOL_ID`
- IAM Role: DynamoDB PutItem, Cognito DescribeUser

`SupplyAggregationLambda`:
- Runtime: Python 3.9
- Memory: 512 MB
- Timeout: 30s
- Environment Variables: `DYNAMODB_TABLE_NAME`
- IAM Role: DynamoDB Query, PutItem

`ForecastLambda`:
- Runtime: Python 3.9
- Memory: 512 MB
- Timeout: 30s
- Environment Variables: `SAGEMAKER_ENDPOINT_NAME`, `DYNAMODB_CACHE_TABLE`
- IAM Role: SageMaker InvokeEndpoint, DynamoDB Query/PutItem

`FairPriceLambda`:
- Runtime: Python 3.9
- Memory: 512 MB
- Timeout: 30s
- Environment Variables: `SAGEMAKER_ENDPOINT_NAME`
- IAM Role: SageMaker InvokeEndpoint, DynamoDB PutItem

`SimulationLambda`:
- Runtime: Python 3.9
- Memory: 512 MB
- Timeout: 30s
- Environment Variables: `DYNAMODB_PRICES_TABLE`, `STORAGE_COST_PER_DAY`
- IAM Role: DynamoDB Query, PutItem

`SMSDeliveryLambda`:
- Runtime: Python 3.9
- Memory: 256 MB
- Timeout: 10s
- Environment Variables: `SNS_TOPIC_ARN`
- IAM Role: SNS Publish, DynamoDB PutItem

**SageMaker Endpoints**:

`agrichain-forecast-endpoint`:
- Model: DeepAR trained model
- Instance Type: ml.m5.large
- Initial Instance Count: 1
- Auto-scaling: Target invocations per instance = 100, scale up to 3 instances
- Data Capture: Enabled for model monitoring

`agrichain-fairprice-endpoint`:
- Model: XGBoost trained model
- Instance Type: ml.m5.large
- Initial Instance Count: 1
- Auto-scaling: Target invocations per instance = 200, scale up to 3 instances

**API Gateway Endpoints**:

```
POST /api/supply
  → InputValidationLambda
  → Store farmer supply data

GET /api/supply/aggregate?fpo_id={id}
  → SupplyAggregationLambda
  → Return aggregated supply by crop

GET /api/forecast?crop={crop}&location={location}
  → ForecastLambda
  → Return 30-day price forecast

POST /api/offers/validate
  → FairPriceLambda
  → Return fair price score

POST /api/simulate
  → SimulationLambda
  → Return scenario comparison

GET /api/dashboard?fpo_id={id}
  → DashboardLambda
  → Return all dashboard data
```

**EventBridge Rules**:

`DailyMandiDataIngestion`:
- Schedule: `cron(0 6 * * ? *)` (6 AM IST daily)
- Target: Glue Job `MandiDataETL`

`WeeklyModelRetraining`:
- Schedule: `cron(0 2 ? * SUN *)` (2 AM Sunday weekly)
- Target: SageMaker Training Job via Lambda

`PriceAlertMonitoring`:
- Schedule: `rate(6 hours)`
- Target: Lambda `PriceMonitorLambda`


## Scalability & Deployment Strategy

### Serverless Architecture Benefits

**Auto-Scaling**:
- Lambda functions scale automatically from 0 to 1000+ concurrent executions
- SageMaker endpoints auto-scale based on invocation rate
- DynamoDB scales read/write capacity on-demand
- No manual capacity planning required

**Cost Efficiency**:
- Pay only for actual usage (no idle server costs)
- Lambda free tier: 1M requests/month, 400,000 GB-seconds/month
- DynamoDB free tier: 25 GB storage, 25 WCU, 25 RCU
- SageMaker endpoints can be shut down when not in use (hackathon demo)

**High Availability**:
- Lambda runs across multiple AZs automatically
- DynamoDB replicates data across 3 AZs
- S3 provides 99.999999999% durability
- CloudFront CDN reduces latency globally

### Horizontal Scalability

**Village to District Scaling**:

**Phase 1: Village Pilot (10 FPOs, 500 farmers)**
- Single AWS region (ap-south-1 Mumbai)
- 1 SageMaker endpoint instance per model
- DynamoDB on-demand mode
- Estimated cost: $200-300/month

**Phase 2: District Expansion (50 FPOs, 5,000 farmers)**
- Same region, auto-scaling enabled
- SageMaker endpoints scale to 2-3 instances during peak hours
- DynamoDB provisioned capacity with auto-scaling
- Estimated cost: $800-1,200/month

**Phase 3: State-Wide (500 FPOs, 50,000 farmers)**
- Multi-region deployment (Mumbai + Hyderabad)
- Route 53 latency-based routing
- DynamoDB Global Tables for cross-region replication
- SageMaker endpoints in both regions
- Estimated cost: $5,000-8,000/month

**Phase 4: National (5,000 FPOs, 500,000 farmers)**
- Multi-region with edge caching (CloudFront)
- SageMaker batch transform for bulk forecasting
- RDS read replicas for analytics
- Dedicated support and enterprise features
- Estimated cost: $30,000-50,000/month

### Multi-Region Deployment Strategy

**Primary Region**: ap-south-1 (Mumbai)
- All services deployed
- Master database (DynamoDB, RDS)
- SageMaker training jobs

**Secondary Region**: ap-south-2 (Hyderabad)
- Read replicas for low-latency access
- SageMaker inference endpoints only
- Failover capability

**Global Services**:
- CloudFront: Edge locations across India
- Route 53: DNS with health checks and failover
- S3: Cross-region replication for model artifacts

**Disaster Recovery**:
- RTO (Recovery Time Objective): 1 hour
- RPO (Recovery Point Objective): 15 minutes
- Automated failover using Route 53 health checks
- Daily snapshots of RDS and DynamoDB backups

### Performance Optimization

**Caching Strategy**:
- CloudFront caches static assets (TTL: 24 hours)
- DynamoDB caches simulation results (TTL: 24 hours)
- Lambda caches SageMaker inference results (TTL: 1 hour)
- API Gateway caches GET responses (TTL: 5 minutes)

**Database Optimization**:
- DynamoDB GSI for efficient queries
- RDS indexes on frequently queried columns (crop_type, location, date)
- Partition key design to avoid hot partitions

**Lambda Optimization**:
- Provisioned concurrency for latency-sensitive functions (ForecastLambda)
- Reuse of SageMaker endpoint connections
- Batch processing for bulk operations

**SageMaker Optimization**:
- Model compilation with SageMaker Neo for faster inference
- Elastic Inference for cost-optimized GPU acceleration (if needed)
- Batch Transform for overnight bulk forecasting


## Security & Compliance Considerations

### IAM Roles and Policies

**Principle of Least Privilege**:
- Each Lambda function has dedicated IAM role with minimal permissions
- SageMaker execution role limited to S3 model bucket and CloudWatch logs
- Glue role limited to source/destination S3 buckets and DynamoDB tables

**Example IAM Policy for ForecastLambda**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": "arn:aws:sagemaker:ap-south-1:*:endpoint/agrichain-forecast-endpoint"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:Query",
        "dynamodb:PutItem"
      ],
      "Resource": "arn:aws:dynamodb:ap-south-1:*:table/Simulations"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:ap-south-1:*:*"
    }
  ]
}
```

**Cross-Service Access**:
- Lambda assumes SageMaker execution role for training jobs
- Glue assumes role to write to S3 and DynamoDB
- API Gateway uses Cognito authorizer for user authentication

### Data Encryption

**Encryption at Rest**:
- S3: Server-side encryption with AWS KMS (SSE-KMS)
- DynamoDB: Encryption enabled by default using AWS-owned keys
- RDS: Encryption enabled at creation using AWS KMS
- SageMaker: Model artifacts encrypted in S3 with KMS
- EBS volumes: Encrypted for Lambda functions in VPC

**Encryption in Transit**:
- API Gateway: HTTPS only (TLS 1.2+)
- CloudFront: HTTPS enforced, HTTP redirects to HTTPS
- SageMaker endpoints: TLS encryption for inference requests
- RDS: SSL/TLS connections enforced
- DynamoDB: HTTPS API calls

**Key Management**:
- AWS KMS customer-managed keys for sensitive data
- Automatic key rotation enabled
- Separate keys for different data classifications (PII vs operational data)

### API Security

**Authentication**:
- Amazon Cognito User Pools for user authentication
- JWT tokens with 1-hour expiration
- Refresh tokens for session management
- MFA optional for FPO leader accounts

**Authorization**:
- Role-based access control (RBAC)
- Roles: `FPO_Leader`, `Farmer`, `Admin`
- API Gateway custom authorizer validates JWT and role
- Fine-grained permissions per endpoint

**Rate Limiting**:
- API Gateway usage plans with throttling
- Burst limit: 100 requests/second
- Steady-state limit: 50 requests/second per user
- Prevents abuse and DDoS attacks

**Input Validation**:
- API Gateway request validation using JSON schemas
- Lambda functions perform additional validation
- Sanitize inputs to prevent injection attacks

**CORS Configuration**:
- Whitelist specific origins (dashboard domain only)
- Restrict HTTP methods (GET, POST only)
- Limit allowed headers

### Data Privacy and Compliance

**PII Protection**:
- Farmer names and phone numbers encrypted at rest
- Access logs exclude PII fields
- Data anonymization for analytics and model training
- Opt-in consent for SMS alerts

**Data Retention**:
- Supply data: 90 days (DynamoDB TTL)
- Price data: 5 years (S3 lifecycle policy)
- Logs: 30 days (CloudWatch retention)
- User accounts: Deleted on request (GDPR compliance)

**Audit Logging**:
- CloudTrail logs all API calls
- DynamoDB streams capture data changes
- Lambda logs all business logic events
- Audit logs retained for 1 year

**Compliance Considerations**:
- GDPR: Right to access, right to deletion, data portability
- India Data Protection Bill: Localization (data stored in India region)
- Agricultural data sovereignty: No cross-border data transfer

### Network Security

**VPC Configuration** (Optional for production):
- Lambda functions in private subnets
- SageMaker endpoints in VPC
- RDS in private subnet with no public access
- NAT Gateway for outbound internet access
- VPC endpoints for AWS services (S3, DynamoDB, SageMaker)

**Security Groups**:
- Restrict inbound traffic to necessary ports only
- RDS security group allows traffic only from Lambda security group
- SageMaker endpoint security group allows traffic only from Lambda

**DDoS Protection**:
- AWS Shield Standard (automatic)
- CloudFront with AWS WAF for advanced protection
- Rate limiting at API Gateway level


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Supply Aggregation Properties

**Property 1: Aggregation Correctness and Traceability**

*For any* set of farmer supply submissions grouped by FPO and crop type, the aggregated quantity should equal the sum of individual farmer quantities, and the breakdown should preserve all individual contributions such that summing the breakdown equals the aggregate.

**Validates: Requirements 1.1, 1.4**

**Property 2: Input Validation for Supply Quantities**

*For any* supply submission, if the quantity is zero, negative, or non-numeric, the system should reject the submission and return an error, leaving the supply data unchanged.

**Validates: Requirements 1.5**

### Price Forecasting Properties

**Property 3: Forecast Output Structure**

*For any* valid forecast request (crop type and location), the system should return exactly 30 daily price predictions, each with confidence intervals (P10, P50, P90 quantiles), and include historical price context for at least the previous 30 days.

**Validates: Requirements 2.1, 2.3, 2.5**

### Collective Selling Simulation Properties

**Property 4: Simulation Completeness**

*For any* simulation request with valid inputs (FPO ID, crop type, quantity), the system should return exactly three scenarios (individual selling, immediate group selling, delayed group selling), each with calculated revenue, and percentage improvement compared to individual selling for the group scenarios.

**Validates: Requirements 3.1, 3.2, 3.6**

**Property 5: Commission Rate Application**

*For any* simulation, the individual selling scenario should apply a commission rate between 8-12%, and both group selling scenarios should apply a commission rate between 3-5%, with group rates always lower than individual rates.

**Validates: Requirements 3.3, 3.4**

**Property 6: Delayed Selling Incorporates Forecasts and Storage Costs**

*For any* delayed selling simulation, the revenue calculation should use forecasted prices (not current prices) and subtract storage costs based on the number of days delayed, such that the optimal day maximizes (forecasted_price × quantity × (1 - commission_rate) - storage_cost).

**Validates: Requirements 3.5**

### Fair Price Validation Properties

**Property 7: Fair Price Score Range**

*For any* buyer offer validation request, the Fair_Price_Score should be a numeric value between 0 and 100 (inclusive).

**Validates: Requirements 4.1**

**Property 8: Score-to-Offer-Ratio Mapping**

*For any* buyer offer, if the offer is within 5% of the estimated fair price (ratio ≥ 0.95), the score should be above 80; if the offer is below 85% of the estimated fair price (ratio < 0.85), the score should be below 50 and include an "unfair" flag.

**Validates: Requirements 4.3, 4.4**

**Property 9: Fair Price Response Completeness**

*For any* fair price validation, the response should include the Fair_Price_Score, the estimated fair price value, and an explanation of factors influencing the estimate.

**Validates: Requirements 4.5, 4.6**

### Logistics Optimization Properties

**Property 10: Logistics Cost Comparison**

*For any* valid origin-destination pair and supply quantity, the system should return both pooled logistics cost per quintal and individual logistics cost per quintal, along with the percentage cost reduction.

**Validates: Requirements 5.1, 5.2, 5.4**

**Property 11: Economies of Scale in Pooled Logistics**

*For any* logistics calculation with quantity greater than 50 quintals, the pooled cost per quintal should be less than the individual cost per quintal, demonstrating economies of scale.

**Validates: Requirements 5.3**

**Property 12: Multi-Vehicle Logistics Calculation**

*For any* supply quantity exceeding truck capacity (200 quintals), the system should calculate costs for multiple trucks, where the number of trucks equals ceiling(quantity / truck_capacity).

**Validates: Requirements 5.6**

### SMS Alert Properties

**Property 13: SMS Message Length Constraint**

*For any* SMS alert generated by the system, the message length should not exceed 160 characters.

**Validates: Requirements 6.3**

**Property 14: SMS Content Completeness**

*For any* SMS alert, the message should contain the crop name, current price (or relevant price information), and a recommended action.

**Validates: Requirements 6.4**

**Property 15: SMS Delivery Logging**

*For any* SMS sent by the system, there should be a corresponding log entry in the Alerts table with recipient phone, message content, timestamp, and delivery status.

**Validates: Requirements 6.6**

**Property 16: Collective Selling Opportunity Alerts**

*For any* simulation result where group selling shows improvement greater than 15% compared to individual selling, the system should trigger an SMS alert to the FPO leader.

**Validates: Requirements 6.2**

### Dashboard Properties

**Property 17: Dashboard Data Completeness**

*For any* dashboard request for a valid FPO ID, the response should include aggregated supply quantities, price forecasts for all tracked crops, recent fair price score calculations, and logistics pooling cost estimates.

**Validates: Requirements 7.1, 7.2, 7.4, 7.5**

**Property 18: CSV Export Validity**

*For any* dashboard data export request, the system should generate a valid CSV file where each row has the same number of columns, headers are present in the first row, and all data is properly escaped.

**Validates: Requirements 7.8**

### Security and Access Control Properties

**Property 19: Authentication Requirement**

*For any* API request to protected endpoints (all endpoints except login/register), if the request does not include a valid authentication token, the system should return a 401 Unauthorized error and not process the request.

**Validates: Requirements 8.3**

**Property 20: Role-Based Access Control**

*For any* authenticated user, if the user has role "Farmer" and attempts to access FPO leader-only endpoints (aggregation, simulation, dashboard), the system should return a 403 Forbidden error; if the user has role "FPO_Leader", they should have access to all FPO-specific data for their FPO only.

**Validates: Requirements 8.4**

**Property 21: Audit Logging for Data Access**

*For any* successful data access operation (read or write to SupplyData, BuyerOffers, Simulations tables), the system should create an audit log entry with user ID, operation type, resource accessed, and timestamp.

**Validates: Requirements 8.5**

**Property 22: Data Deletion on Request**

*For any* user deletion request, the system should remove all personal data (name, phone, supply records) associated with that user from all tables within the retention policy timeframe.

**Validates: Requirements 8.7**

### Model Transparency Properties

**Property 23: Forecast Explainability**

*For any* price forecast response, the system should include key factors influencing the prediction (such as historical trends, seasonality indicators) and model accuracy metrics (RMSE, MAE).

**Validates: Requirements 9.1, 9.4**

**Property 24: Fair Price Estimation Transparency**

*For any* fair price estimation, the response should list the data sources used (e.g., "Agmarknet historical data") and the key variables in the model (e.g., crop type, location, seasonality).

**Validates: Requirements 9.2**

**Property 25: Simulation Assumptions Documentation**

*For any* simulation result, the response should include documentation of assumptions used in calculations (commission rates, storage costs, forecast source).

**Validates: Requirements 9.5**


## Error Handling

### Error Categories

**Input Validation Errors**:
- Invalid crop type (not in supported list)
- Invalid quantity (zero, negative, non-numeric)
- Invalid location (empty, malformed)
- Invalid date format
- Missing required fields

**Authentication/Authorization Errors**:
- Missing or expired JWT token
- Invalid credentials
- Insufficient permissions for requested operation
- User not found

**Business Logic Errors**:
- Insufficient historical data for forecast (< 24 months)
- No supply data available for aggregation
- Buyer offer validation requested for unsupported crop
- Simulation requested with invalid parameters

**External Service Errors**:
- SageMaker endpoint unavailable or timeout
- DynamoDB throttling or service error
- S3 access denied or object not found
- SNS SMS delivery failure

**System Errors**:
- Lambda timeout (> 30 seconds)
- Out of memory error
- Unexpected exception in business logic

### Error Response Format

All API errors follow consistent JSON structure:

```json
{
  "error": {
    "code": "INVALID_QUANTITY",
    "message": "Supply quantity must be a positive number",
    "details": {
      "field": "quantity",
      "provided_value": "-10",
      "constraint": "quantity > 0"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "abc123-def456"
  }
}
```

### Error Handling Strategies

**Graceful Degradation**:
- If SageMaker forecast endpoint is unavailable, return cached forecast with warning
- If fair price model fails, return historical average with lower confidence score
- If SMS delivery fails, log error and retry up to 3 times with exponential backoff

**Retry Logic**:
- Transient errors (throttling, timeouts): Retry with exponential backoff (1s, 2s, 4s)
- Maximum 3 retry attempts
- Circuit breaker pattern for repeated failures (open circuit after 5 consecutive failures)

**Fallback Mechanisms**:
- Forecast fallback: Use simple moving average if ML model fails
- Fair price fallback: Use 30-day historical average if regression model fails
- Simulation fallback: Use current prices only if forecast unavailable

**User-Friendly Error Messages**:
- Technical errors translated to plain language
- Actionable guidance provided (e.g., "Please check your internet connection and try again")
- Contact information for support (for system errors)

### Logging and Monitoring

**CloudWatch Alarms**:
- Lambda error rate > 5%
- SageMaker endpoint latency > 5 seconds
- DynamoDB throttled requests > 10/minute
- SMS delivery failure rate > 10%

**Error Metrics**:
- Error count by error code
- Error rate by API endpoint
- Mean time to recovery (MTTR)
- Error distribution by user role

**Alerting**:
- Critical errors (system down): Immediate SNS notification to admin
- High error rate: Email alert within 15 minutes
- Degraded performance: Dashboard warning indicator


## Testing Strategy

### Dual Testing Approach

The AgriChain Intelligence system requires both unit testing and property-based testing for comprehensive coverage. These approaches are complementary:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs

Together, they provide comprehensive coverage where unit tests catch concrete bugs and property tests verify general correctness.

### Property-Based Testing

**Framework Selection**:
- **Python**: Hypothesis library
- **JavaScript/TypeScript**: fast-check library
- **Configuration**: Minimum 100 iterations per property test (due to randomization)

**Property Test Implementation**:

Each correctness property from the design document must be implemented as a property-based test. Each test must be tagged with a comment referencing the design property:

```python
# Feature: agrichain-intelligence, Property 1: Aggregation Correctness and Traceability
@given(st.lists(st.tuples(st.text(), st.integers(min_value=1, max_value=1000))))
def test_supply_aggregation_correctness(farmer_supplies):
    # Generate random farmer supply data
    # Call aggregation function
    # Verify sum of individual quantities equals aggregate
    # Verify breakdown preserves all contributions
    pass
```

**Property Test Coverage**:
- Property 1-2: Supply aggregation and validation
- Property 3: Forecast output structure
- Property 4-6: Simulation completeness and calculations
- Property 7-9: Fair price scoring and validation
- Property 10-12: Logistics cost calculations
- Property 13-16: SMS alert generation and logging
- Property 17-18: Dashboard data completeness and export
- Property 19-22: Security and access control
- Property 23-25: Model transparency and explainability

**Generator Strategies**:
- Crop types: Sample from predefined list of 15 crops
- Quantities: Integers between 1 and 10,000 quintals
- Prices: Floats between 500 and 5,000 rupees per quintal
- Locations: Sample from list of major mandi locations
- Dates: Valid dates within last 5 years
- FPO IDs: UUIDs or integers 1-1000

### Unit Testing

**Unit Test Focus Areas**:

**Specific Examples**:
- Test aggregation with exactly 3 farmers, 2 crop types
- Test forecast for wheat in Punjab on specific date
- Test simulation with 1000 quintals at ₹1200/quintal
- Test fair price validation with offer exactly at fair price

**Edge Cases**:
- Empty supply data (no farmers)
- Single farmer (no aggregation needed)
- Quantity exactly at truck capacity (200 quintals)
- Buyer offer exactly at scoring threshold (95%, 85%)
- SMS message exactly 160 characters
- Forecast request for crop with minimal historical data

**Error Conditions**:
- Invalid crop type (not in supported list)
- Negative quantity
- Missing required fields
- Expired authentication token
- Insufficient permissions
- SageMaker endpoint timeout
- DynamoDB throttling

**Integration Points**:
- Lambda to DynamoDB integration
- Lambda to SageMaker endpoint integration
- API Gateway to Lambda integration
- SNS SMS delivery
- S3 file upload/download

**Unit Test Examples**:

```python
def test_aggregation_empty_supply():
    """Test aggregation with no supply data returns empty result"""
    result = aggregate_supply(fpo_id="123", supply_data=[])
    assert result == {}

def test_fair_price_score_at_threshold():
    """Test fair price score exactly at 95% threshold"""
    score = calculate_fair_price_score(buyer_offer=950, fair_price=1000)
    assert score >= 80

def test_sms_length_exactly_160():
    """Test SMS generation with content that results in exactly 160 chars"""
    message = generate_sms_alert(crop="wheat", price=1250, action="sell")
    assert len(message) == 160

def test_invalid_crop_type_error():
    """Test that invalid crop type raises appropriate error"""
    with pytest.raises(InvalidCropTypeError):
        request_forecast(crop="invalid_crop", location="punjab")
```

### Integration Testing

**End-to-End Flows**:
1. Farmer supply submission → Aggregation → Dashboard display
2. Forecast request → SageMaker inference → Cache → Response
3. Buyer offer → Fair price validation → SMS alert (if unfair)
4. Simulation request → Forecast retrieval → Calculation → Response
5. Price update → Alert trigger → SMS delivery → Logging

**AWS Service Integration**:
- API Gateway → Lambda (with actual HTTP requests)
- Lambda → DynamoDB (with local DynamoDB or test tables)
- Lambda → SageMaker (with mock endpoint or test endpoint)
- Lambda → SNS (with test phone numbers or mock)

**Test Environment**:
- Separate AWS account or isolated resources
- Test DynamoDB tables with "test-" prefix
- Test SageMaker endpoints with smaller instance types
- Test SNS topic with admin phone numbers only

### Performance Testing

**Load Testing**:
- Simulate 100 concurrent FPO leader users
- Test API response times under load
- Verify auto-scaling behavior
- Measure SageMaker endpoint latency

**Stress Testing**:
- Test with 10,000 farmer records
- Test aggregation with 20+ crop types
- Test batch forecasting for all crops
- Test SMS delivery to 1,000 recipients

**Tools**:
- Apache JMeter or Locust for load testing
- AWS CloudWatch for monitoring
- SageMaker Model Monitor for inference metrics

### Test Data Management

**Synthetic Data Generation**:
- Generate realistic farmer profiles (500-1000 records)
- Generate historical mandi prices (24 months, 15 crops, 50 locations)
- Generate buyer offers with known fair/unfair labels
- Generate supply submissions with various patterns

**Test Data Storage**:
- S3 bucket: `agrichain-test-data`
- Seed data scripts for reproducible test environments
- Anonymized production data (if available) for realistic testing

### Continuous Integration

**CI/CD Pipeline**:
1. Code commit triggers AWS CodePipeline
2. Run unit tests (pytest or jest)
3. Run property tests (Hypothesis or fast-check)
4. Run linting and code quality checks
5. Deploy to test environment
6. Run integration tests
7. If all pass, deploy to staging
8. Manual approval for production deployment

**Test Coverage Goals**:
- Unit test coverage: > 80%
- Property test coverage: All 25 properties implemented
- Integration test coverage: All critical flows
- End-to-end test coverage: All user journeys


## Limitations & Future Enhancements

### Current Limitations

**Data Limitations**:
- Relies on historical Agmarknet data which may have gaps or delays
- No real-time integration with live mandi trading systems
- Synthetic farmer data for hackathon (not real FPO data)
- Limited to 15 major crop types (excludes specialty crops)
- No integration with weather data or crop yield predictions

**Model Limitations**:
- Price forecasts cannot account for unpredictable events (droughts, policy changes, pandemics)
- Fair price estimates are indicative, not guaranteed market prices
- Models trained on historical data may not capture emerging market dynamics
- No demand-side modeling (only supply-side focus)
- Limited to Indian agricultural markets (not generalizable globally)

**System Limitations**:
- SMS delivery depends on third-party telecom infrastructure reliability
- Dashboard requires internet connectivity (limited offline capability)
- No mobile app (web-only interface)
- Simulation results are estimates, not binding price guarantees
- No automated trading or transaction execution
- Limited to English and Hindi languages

**Scalability Limitations**:
- SageMaker endpoints have cold start latency (~5 seconds)
- DynamoDB query patterns may need optimization for very large FPOs (>10,000 farmers)
- SMS costs scale linearly with user base
- No multi-tenancy isolation (all FPOs share same infrastructure)

### Future Enhancements

**Phase 2: Real-Time Market Integration**

- **Live Mandi Price Feeds**: Integrate with real-time mandi price APIs for up-to-the-minute data
- **Automated Price Alerts**: Trigger alerts within seconds of significant price movements
- **Bid/Ask Matching**: Connect FPOs directly with verified buyers for transparent negotiations
- **Transaction Execution**: Enable digital contracts and payment processing through the platform

**Phase 3: Advanced AI Capabilities**

- **Demand Forecasting**: Predict buyer demand patterns to optimize selling timing
- **Reinforcement Learning for Negotiation**: AI agent that learns optimal negotiation strategies
- **Computer Vision for Quality Assessment**: Use smartphone photos to assess crop quality and adjust price estimates
- **NLP for Market Sentiment**: Analyze news and social media to detect market sentiment shifts
- **Multi-Modal Models**: Combine price data, weather data, satellite imagery for holistic predictions

**Phase 4: Ecosystem Integration**

- **FPO ERP Integration**: Connect with existing FPO management systems (accounting, inventory)
- **Bank Integration**: Direct integration with agricultural lending platforms for credit access
- **Insurance Integration**: Connect with crop insurance providers for risk management
- **Logistics Marketplace**: Platform for FPOs to book shared transportation with verified logistics providers
- **Buyer Marketplace**: Verified buyer network with ratings and transaction history

**Phase 5: Expanded Coverage**

- **Multi-Crop Bundling**: Optimize selling strategies for FPOs with diverse crop portfolios
- **Cross-Regional Arbitrage**: Identify price differences across regions for strategic selling
- **Export Market Intelligence**: Extend to international markets and export opportunities
- **Specialty Crops**: Add support for fruits, vegetables, spices beyond major staples
- **Livestock and Dairy**: Extend platform to animal husbandry and dairy cooperatives

**Phase 6: Advanced Analytics**

- **Predictive Analytics Dashboard**: Forecast FPO revenue, identify at-risk farmers, optimize crop mix
- **Benchmarking**: Compare FPO performance against regional and national averages
- **Impact Measurement**: Track actual revenue improvements and farmer welfare outcomes
- **Policy Simulation**: Model impact of government policies (MSP changes, subsidies) on farmer income
- **Climate Risk Modeling**: Assess climate change impact on crop prices and recommend adaptation strategies

**Phase 7: Mobile and Offline Capabilities**

- **Native Mobile Apps**: iOS and Android apps with offline-first architecture
- **USSD Interface**: Feature phone support via USSD codes (no smartphone required)
- **Voice Interface**: IVR system for voice-based market intelligence (regional languages)
- **Progressive Web App**: Enhanced PWA with service workers for offline caching
- **Bluetooth Mesh Networking**: Peer-to-peer data sync in low-connectivity areas

**Phase 8: Blockchain and Traceability**

- **Supply Chain Traceability**: Track produce from farm to consumer using blockchain
- **Smart Contracts**: Automated payment release upon delivery verification
- **Tokenization**: Digital tokens representing crop ownership for fractional trading
- **Transparent Pricing**: Immutable record of all price negotiations and transactions
- **Fair Trade Certification**: Automated verification of fair trade practices

**Phase 9: Social and Collaborative Features**

- **Farmer Forums**: Community platform for knowledge sharing and peer support
- **Expert Network**: Connect farmers with agricultural extension officers and agronomists
- **Cooperative Formation**: Tools to help farmers organize into new FPOs
- **Training Modules**: Video tutorials on collective bargaining, market intelligence, digital literacy
- **Gamification**: Rewards and recognition for farmers who adopt best practices

**Phase 10: Sustainability and ESG**

- **Carbon Footprint Tracking**: Calculate and offset carbon emissions from agricultural activities
- **Sustainable Farming Incentives**: Premium pricing for organic and regenerative agriculture
- **Water Usage Optimization**: Recommendations for efficient irrigation based on crop and weather
- **Biodiversity Monitoring**: Track and reward practices that enhance biodiversity
- **ESG Reporting**: Generate sustainability reports for FPOs seeking impact investment

### Technology Evolution

**Infrastructure Improvements**:
- Migrate to serverless containers (AWS Fargate) for more complex workloads
- Implement GraphQL API for more flexible data queries
- Add Redis/ElastiCache for sub-second response times
- Use AWS AppSync for real-time data synchronization
- Implement AWS IoT Core for sensor data from farms (soil moisture, weather stations)

**AI/ML Improvements**:
- Upgrade to transformer-based models for better time-series forecasting
- Implement federated learning for privacy-preserving model training across FPOs
- Use AutoML (SageMaker Autopilot) for continuous model optimization
- Implement A/B testing framework for model experimentation
- Add explainable AI (SHAP, LIME) for deeper model interpretability

**Security Enhancements**:
- Implement zero-trust architecture with AWS PrivateLink
- Add biometric authentication for high-value transactions
- Implement blockchain-based audit trails for compliance
- Add anomaly detection for fraud prevention
- Implement data residency controls for international expansion

### Hackathon to Production Roadmap

**Hackathon MVP** (Current):
- Core features: Aggregation, forecasting, simulation, fair price validation
- Synthetic data and public datasets
- Single AWS region deployment
- Basic web dashboard and SMS alerts

**Pilot Deployment** (3 months):
- Partner with 5-10 FPOs in one district
- Real farmer data and feedback
- Enhanced UI/UX based on user testing
- Integration with local mandi data sources
- Performance optimization and bug fixes

**Regional Rollout** (6-12 months):
- Expand to 50-100 FPOs across 2-3 states
- Multi-language support (5+ regional languages)
- Mobile app launch
- Integration with state agricultural departments
- Partnerships with banks and insurance providers

**National Scale** (12-24 months):
- 1,000+ FPOs across India
- Advanced AI features (demand forecasting, RL negotiation)
- Ecosystem integrations (ERP, logistics, buyers)
- Impact measurement and reporting
- Sustainability and ESG features

**International Expansion** (24+ months):
- Adapt to other developing countries (Africa, Southeast Asia, Latin America)
- Multi-currency and multi-market support
- Localization for different agricultural systems
- Partnerships with international development organizations
- Open-source community edition for NGOs and governments

