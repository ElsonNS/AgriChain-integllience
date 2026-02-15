# Requirements Document: AgriChain Intelligence

## Introduction

AgriChain Intelligence is an AI-powered market intelligence system designed to empower rural producer groups (Farmer Producer Organizations, cooperatives, and farmer clusters) with data-driven insights for collective bargaining and fair market access. The system addresses critical information asymmetry in agricultural markets by providing price forecasting, fair price validation, collective selling simulations, and logistics optimization to help small-scale farmers maximize their returns through coordinated group action.

## Project Overview

### Problem Description

Small-scale farmers in rural India face significant challenges in agricultural markets:
- Information asymmetry in mandi (wholesale market) pricing leads to exploitation by intermediaries
- Individual farmers have weak bargaining power when negotiating with buyers
- Lack of visibility into fair market prices results in undervaluation of produce
- Fragmented selling and inefficient logistics reduce profit margins
- Limited access to technology and low bandwidth connectivity restrict access to market intelligence

### Objective

The AgriChain Intelligence system aims to level the playing field by providing FPO leaders and cooperative managers with AI-driven tools to:
- Aggregate supply across member farmers for stronger negotiating positions
- Forecast market prices using historical data and trends
- Validate buyer offers against fair market prices
- Simulate collective selling scenarios to demonstrate value of group action
- Optimize logistics costs through pooled transportation
- Deliver actionable insights via low-bandwidth channels (SMS, lightweight dashboards)

### Target Users

- FPO (Farmer Producer Organization) leaders and managers
- Agricultural cooperative administrators
- Rural producer group coordinators
- Individual farmers within organized groups

## Problem Statement

### Information Asymmetry

Farmers lack access to real-time and historical market price data, making them vulnerable to price manipulation by intermediaries and traders who possess superior market knowledge.

### Weak Bargaining Power

Individual small-scale farmers selling limited quantities have minimal leverage in price negotiations, often accepting below-market rates due to lack of alternatives.

### Pricing Visibility Gap

Without transparent benchmarks for fair pricing, farmers cannot assess whether buyer offers represent genuine market value or exploitative pricing.

### Logistics Inefficiency

Fragmented selling by individual farmers results in higher per-unit transportation costs and missed opportunities for bulk logistics optimization.

## Glossary

- **System**: The AgriChain Intelligence platform
- **FPO**: Farmer Producer Organization
- **Mandi**: Traditional wholesale agricultural market in India
- **Producer_Group**: Collection of farmers organized as FPO, cooperative, or cluster
- **Fair_Price_Score**: Confidence metric indicating alignment between buyer offer and estimated fair market price
- **Collective_Selling_Simulation**: Scenario analysis comparing individual vs group selling outcomes
- **Supply_Aggregation**: Process of combining produce quantities from multiple farmers
- **Price_Forecasting_Model**: Time-series ML model predicting future mandi prices
- **Logistics_Pool**: Shared transportation arrangement for multiple farmers
- **Dashboard**: Web-based interface for FPO leaders to access insights
- **SMS_Alert**: Text message notification for low-bandwidth users
- **Agmarknet**: Government of India's agricultural marketing information network

## Requirements

### Requirement 1: Supply Aggregation

**User Story:** As an FPO leader, I want to aggregate supply quantities across member farmers, so that I can present consolidated volumes to buyers and strengthen our bargaining position.

#### Acceptance Criteria

1. WHEN an FPO leader inputs farmer supply data, THE System SHALL calculate total aggregated quantity by crop type
2. WHEN supply data is updated, THE System SHALL recalculate aggregated quantities within 2 seconds
3. THE System SHALL support aggregation for at least 20 different crop types simultaneously
4. WHEN displaying aggregated supply, THE System SHALL show breakdown by individual farmer contributions
5. THE System SHALL validate that individual supply quantities are positive non-zero values

### Requirement 2: Time-Series Price Forecasting

**User Story:** As an FPO leader, I want to forecast future mandi prices for my crops, so that I can time our collective sales to maximize returns.

#### Acceptance Criteria

1. WHEN an FPO leader requests price forecast for a crop, THE System SHALL generate predictions for the next 30 days
2. THE Price_Forecasting_Model SHALL use historical mandi price data from at least the previous 24 months
3. WHEN generating forecasts, THE System SHALL provide confidence intervals for predicted prices
4. THE System SHALL update forecast models weekly using latest available mandi data
5. WHEN displaying forecasts, THE System SHALL show historical price trends for context
6. THE System SHALL support price forecasting for at least 15 major crop types

### Requirement 3: Collective Selling Simulation

**User Story:** As an FPO leader, I want to simulate different selling scenarios, so that I can demonstrate to farmers the financial benefits of collective action.

#### Acceptance Criteria

1. WHEN an FPO leader initiates simulation, THE System SHALL compare three scenarios: individual selling, immediate group selling, and delayed group selling
2. THE Collective_Selling_Simulation SHALL calculate estimated revenue for each scenario based on current and forecasted prices
3. WHEN simulating individual selling, THE System SHALL apply typical intermediary commission rates of 8-12%
4. WHEN simulating group selling, THE System SHALL apply reduced commission rates of 3-5% due to bulk volumes
5. WHEN simulating delayed selling, THE System SHALL use forecasted prices and storage cost assumptions
6. THE System SHALL display percentage revenue improvement for group scenarios compared to individual selling
7. THE System SHALL generate simulation results within 5 seconds of request

### Requirement 4: Fair Price Confidence Score

**User Story:** As an FPO leader, I want to validate buyer offers against fair market prices, so that I can negotiate confidently and avoid exploitation.

#### Acceptance Criteria

1. WHEN a buyer offer is entered, THE System SHALL calculate a Fair_Price_Score between 0 and 100
2. THE System SHALL estimate fair market price using regression model trained on historical mandi data
3. WHEN the buyer offer is within 5% of estimated fair price, THE System SHALL assign Fair_Price_Score above 80
4. WHEN the buyer offer is below 85% of estimated fair price, THE System SHALL assign Fair_Price_Score below 50 and flag as potentially unfair
5. THE System SHALL display the estimated fair price range alongside the Fair_Price_Score
6. THE System SHALL provide explanation of factors influencing the fair price estimate
7. THE System SHALL update fair price estimates daily using latest mandi data

### Requirement 5: Logistics Pooling Cost Estimation

**User Story:** As an FPO leader, I want to estimate transportation costs for pooled logistics, so that I can demonstrate cost savings to member farmers.

#### Acceptance Criteria

1. WHEN an FPO leader inputs origin and destination locations, THE System SHALL estimate transportation cost per quintal for pooled shipment
2. THE System SHALL compare pooled logistics cost against individual farmer transportation costs
3. WHEN calculating pooled costs, THE System SHALL apply economies of scale for bulk transportation
4. THE System SHALL display percentage cost reduction achieved through logistics pooling
5. THE System SHALL support distance-based cost calculation for routes up to 500 kilometers
6. WHEN supply quantity exceeds truck capacity, THE System SHALL calculate multi-vehicle logistics costs

### Requirement 6: SMS-Based Low-Bandwidth Alerts

**User Story:** As a farmer with limited internet access, I want to receive critical market alerts via SMS, so that I can stay informed without requiring smartphone or data connection.

#### Acceptance Criteria

1. WHEN significant price movements occur (>10% change), THE System SHALL send SMS_Alert to registered farmers within 1 hour
2. WHEN a collective selling opportunity is identified, THE System SHALL send SMS_Alert to FPO leader
3. THE System SHALL limit SMS message length to 160 characters for compatibility
4. WHEN sending SMS_Alert, THE System SHALL include crop name, current price, and recommended action
5. THE System SHALL support SMS delivery to at least 1000 registered users
6. THE System SHALL log all SMS_Alert deliveries for audit purposes

### Requirement 7: Dashboard for FPO Leaders

**User Story:** As an FPO leader, I want a centralized dashboard to access all market intelligence features, so that I can efficiently manage collective selling decisions.

#### Acceptance Criteria

1. THE Dashboard SHALL display current aggregated supply quantities for all member farmers
2. THE Dashboard SHALL show price forecasts for all tracked crop types
3. THE Dashboard SHALL provide access to collective selling simulation tool
4. THE Dashboard SHALL display recent Fair_Price_Score calculations for buyer offers
5. THE Dashboard SHALL show logistics pooling cost estimates
6. WHEN accessing the Dashboard on mobile devices, THE System SHALL render responsive layout optimized for small screens
7. THE Dashboard SHALL load initial view within 3 seconds on 3G network connection
8. THE Dashboard SHALL support data export to CSV format for offline analysis

### Requirement 8: Data Security and Privacy

**User Story:** As an FPO leader, I want farmer data to be securely stored and protected, so that sensitive business information remains confidential.

#### Acceptance Criteria

1. THE System SHALL encrypt all farmer supply data at rest using AES-256 encryption
2. THE System SHALL encrypt all data transmissions using TLS 1.2 or higher
3. WHEN a user logs in, THE System SHALL require authentication via username and password
4. THE System SHALL implement role-based access control with separate permissions for FPO leaders and farmers
5. THE System SHALL log all data access attempts for security audit
6. THE System SHALL automatically log out inactive users after 15 minutes
7. THE System SHALL comply with data retention policies requiring deletion of personal data upon user request

### Requirement 9: Model Transparency and Explainability

**User Story:** As an FPO leader, I want to understand how AI models generate predictions, so that I can trust the recommendations and explain them to farmers.

#### Acceptance Criteria

1. WHEN displaying price forecasts, THE System SHALL show key factors influencing the prediction
2. WHEN calculating Fair_Price_Score, THE System SHALL list the data sources and variables used in estimation
3. THE System SHALL provide plain-language explanations of model outputs suitable for non-technical users
4. THE System SHALL display model accuracy metrics (RMSE, MAE) for price forecasting
5. WHEN simulation results are shown, THE System SHALL explain assumptions used in calculations
6. THE System SHALL provide access to historical model performance data

### Requirement 10: Scalability

**User Story:** As a system administrator, I want the platform to scale from village-level to district-level usage, so that it can serve growing numbers of FPOs and farmers.

#### Acceptance Criteria

1. THE System SHALL support at least 100 concurrent FPO leader users without performance degradation
2. THE System SHALL handle supply aggregation for at least 10,000 individual farmers
3. WHEN user load increases by 50%, THE System SHALL maintain response times within 10% of baseline
4. THE System SHALL support data storage for at least 5 years of historical mandi prices
5. THE System SHALL process batch price forecast updates for 15 crop types within 30 minutes

## Non-Functional Requirements

### Performance

- Price forecast generation: Maximum 5 seconds per crop
- Fair Price Score calculation: Maximum 2 seconds per offer
- Dashboard initial load: Maximum 3 seconds on 3G connection
- Collective selling simulation: Maximum 5 seconds per scenario set
- Supply aggregation recalculation: Maximum 2 seconds

### Scalability

- Support 100 concurrent FPO leader users
- Handle 10,000 individual farmer records
- Store 5 years of historical mandi price data
- Process 15 crop types simultaneously
- Scale to district-level deployment (50+ FPOs)

### Availability

- System uptime: 99% during market hours (6 AM - 8 PM IST)
- Scheduled maintenance windows: Maximum 4 hours per month
- SMS alert delivery: 95% success rate within 1 hour

### Low-Bandwidth Compatibility

- Dashboard optimized for 3G network speeds
- SMS alerts as primary notification channel
- Minimal data payload for mobile dashboard (<500 KB per page load)
- Progressive web app support for offline access to cached data

### Data Security

- AES-256 encryption for data at rest
- TLS 1.2+ for data in transit
- Role-based access control (RBAC)
- Automatic session timeout after 15 minutes
- Audit logging for all data access
- GDPR-compliant data deletion on request

### Model Transparency

- Plain-language explanations for all AI outputs
- Display of model accuracy metrics (RMSE, MAE, R²)
- Visibility into data sources and variables used
- Historical model performance tracking
- Assumption documentation for simulations

### Usability

- Dashboard interface in English and Hindi
- Mobile-responsive design for smartphone access
- Maximum 3 clicks to access any major feature
- Contextual help text for all AI-generated insights
- CSV export capability for offline analysis

## AI/ML Requirements

### Time-Series Forecasting Model

- Algorithm: ARIMA, Prophet, or LSTM-based time-series model
- Training data: Minimum 24 months of historical mandi prices
- Forecast horizon: 30 days ahead
- Update frequency: Weekly retraining with latest data
- Performance target: MAPE (Mean Absolute Percentage Error) < 15%
- Output: Daily price predictions with 95% confidence intervals

### Fair Price Regression Model

- Algorithm: Random Forest, Gradient Boosting, or Linear Regression
- Features: Historical mandi prices, seasonality, crop type, location, supply volume
- Training data: Historical mandi transactions from Agmarknet
- Performance target: R² > 0.75 on validation set
- Output: Estimated fair price with confidence score (0-100)

### Scenario Simulation Engine

- Rule-based calculation engine (not ML-based)
- Inputs: Current prices, forecasted prices, supply quantities, commission rates, logistics costs
- Scenarios: Individual selling, immediate group selling, delayed group selling
- Output: Revenue estimates and percentage improvements for each scenario

### Optional: Supply Clustering

- Algorithm: K-means or hierarchical clustering
- Purpose: Group farmers by crop type, location, and harvest timing
- Use case: Identify optimal sub-groups for coordinated selling
- Implementation: Phase 2 enhancement (not required for MVP)

## Data Requirements

### Historical Mandi Price Data

- Source: Agmarknet (Government of India public dataset)
- Coverage: Minimum 24 months of daily price data
- Crop types: At least 15 major crops (wheat, rice, cotton, sugarcane, etc.)
- Geographic scope: Major mandis in target states
- Format: CSV or API access
- Update frequency: Daily or weekly

### Farmer and FPO Data

- Source: Synthetic data for hackathon (simulated farmer groups)
- Fields: Farmer ID, FPO membership, crop type, supply quantity, location
- Volume: 500-1000 synthetic farmer records across 10-20 FPOs
- Privacy: No real farmer PII used in hackathon prototype

### Logistics Cost Data

- Source: Assumptions based on typical transportation rates
- Parameters: Cost per kilometer, vehicle capacity, loading/unloading fees
- Basis: Industry averages for agricultural logistics in India

### Data Storage

- Database: PostgreSQL or MongoDB for structured data
- Time-series storage: InfluxDB or TimescaleDB for price history
- File storage: S3 or equivalent for CSV exports and model artifacts
- Estimated storage: 5 GB for 5 years of mandi data + farmer records

## Assumptions & Limitations

### Data Assumptions

- Historical mandi price data from Agmarknet is accurate and representative
- Synthetic farmer data adequately simulates real-world FPO structures
- Transportation cost assumptions reflect typical market rates
- Commission rate assumptions (8-12% individual, 3-5% group) are realistic

### Model Limitations

- Price forecasting accuracy depends on data quality and market stability
- Forecasts cannot account for unpredictable events (weather disasters, policy changes)
- Fair price estimates are indicative, not guaranteed market prices
- Models trained on historical data may not capture emerging market dynamics

### System Limitations

- No real-time integration with live mandi trading systems
- SMS delivery depends on third-party telecom infrastructure
- Simulation results are estimates, not binding price guarantees
- System provides decision support, not automated trading execution

### Hackathon Scope

- Prototype uses public and synthetic datasets (no proprietary data)
- Focus on core features (supply aggregation, forecasting, simulation, fair price scoring)
- Advanced features (clustering, multi-language NLP) are optional enhancements
- Deployment limited to demo environment (not production-ready)

## Success Metrics

### Revenue Impact Simulation

- Target: Demonstrate 15-25% estimated revenue improvement through collective selling
- Measurement: Average percentage gain across simulated scenarios
- Validation: Compare simulation outputs against historical FPO performance data (if available)

### Fair Price Detection Accuracy

- Target: Fair Price Score correctly identifies unfair offers (>15% below market) with 85% accuracy
- Measurement: Precision and recall on validation dataset of known fair/unfair offers
- Validation: Expert review of flagged offers by agricultural economists

### Logistics Cost Reduction

- Target: Demonstrate 20-30% transportation cost reduction through logistics pooling
- Measurement: Percentage savings in simulated pooled vs individual logistics scenarios
- Validation: Compare against actual FPO logistics cost data (if available)

### User Adoption and Engagement

- Target: 80% of demo FPO leaders successfully complete end-to-end workflow (supply entry → simulation → decision)
- Measurement: User interaction logs and task completion rates during hackathon demo
- Validation: User feedback surveys and usability testing

### Model Performance

- Price forecasting MAPE: < 15%
- Fair price regression R²: > 0.75
- Dashboard load time: < 3 seconds on 3G
- SMS delivery success rate: > 95%

### Hackathon Evaluation Criteria

- Innovation: Novel application of AI for rural agricultural markets
- Impact: Potential to improve farmer livelihoods and market fairness
- Feasibility: Realistic implementation using available data and technology
- Scalability: Ability to expand from pilot to district/state level
- AWS Integration: Effective use of AWS services (SageMaker, Lambda, S3, SNS, etc.)
