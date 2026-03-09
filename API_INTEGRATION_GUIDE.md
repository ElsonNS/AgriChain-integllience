# AgriChain API Integration Guide

## Overview
This document describes the complete API integration between the AgriChain React frontend and AWS Lambda backend. The backend uses AI techniques like linear regression and statistical models implemented directly within Lambda functions for real-time price prediction and risk analysis.

## Architecture

### Frontend
- **Framework**: React + TypeScript + Vite
- **Location**: `frontend/dashboard/`
- **API Client**: Axios with interceptors for error handling

### Backend
- **Platform**: AWS Lambda + API Gateway
- **Base URL**: `your url`
- **Architecture**: Serverless

## Environment Configuration

### Environment Variables
Create a `.env` file in `frontend/dashboard/`:

```env
VITE_API_URL=your url
VITE_ENV=production
```

## API Endpoints

### 1. Price Prediction - POST /predict
**Lambda Function**: `forecast-lambda`

**Request**:
```json
{
  "crop": "Tomato",
  "mandi": "Kolar",
  "timestamp": "2026-03-09T16:54:00.546Z"
}
```

**Response**:
```json
{
  "predictedPrice": 45.50,
  "confidence": 87.5,
  "trend": "up",
  "recommendation": "Consider holding for better prices"
}
```

**Frontend Usage**: `Dashboard.tsx` - AI Price Prediction feature

---

### 2. Price Forecast - POST /predict
**Lambda Function**: `forecast-lambda`

**Request**:
```json
{
  "crop": "Tomato",
  "mandi": "Kolar",
  "forecast": true,
  "days": 14
}
```

**Response**:
```json
{
  "forecast": [
    {
      "date": "2026-03-01",
      "price": 42.30
    },
    {
      "date": "2026-03-02",
      "price": 43.10
    }
  ]
}
```

**Frontend Usage**: `Dashboard.tsx` - Historical & Forecast Prices chart

---

### 3. Supply/Demand Data - POST /supply
**Lambda Function**: `supply-lambda`

**Request**:
```json
{
  "crop": "Tomato",
  "mandi": "Kolar"
}
```

**Response**:
```json
{
  "supply": [50, 55, 60, 58, 62, 65, 70],
  "demand": [45, 48, 52, 55, 58, 60, 65]
}
```

**Frontend Usage**: `Dashboard.tsx` - Supply vs Demand chart

---

### 4. Market Insights - POST /supply
**Lambda Function**: `supply-lambda`

**Request**:
```json
{
  "crop": "Tomato",
  "allMandis": true
}
```

**Response**:
```json
{
  "markets": [
    {
      "mandi": "Kolar",
      "price": 45.50,
      "trend": "up",
      "action": "Monitor"
    },
    {
      "mandi": "Bangalore",
      "price": 42.30,
      "trend": "down",
      "action": "Sell"
    }
  ]
}
```

**Frontend Usage**: `MarketInsights.tsx` - Multi-mandi comparison table

---

### 5. Risk Simulation - GET /simulate
**Lambda Function**: `simulation-lambda`

**Request** (Query Parameters):
```
?crop=Tomato&currentPrice=45&rainfall=Normal&transportCost=5
```

**Response**:
```json
{
  "priceRange": {
    "min": 40.50,
    "max": 50.75
  },
  "riskLevel": "medium",
  "recommendation": "Monitor market conditions closely"
}
```

**Frontend Usage**: `RiskSimulation.tsx` - Risk assessment and scenario analysis

---

### 6. SMS Alert Subscription - GET /notify
**Lambda Function**: `send-sms-alert`

**Request** (Query Parameters):
```
?phoneNumber=+919876543210&crop=Tomato&mandi=Kolar
```

**Response**:
```json
{
  "success": true,
  "message": "Successfully subscribed to alerts"
}
```

**Frontend Usage**: `Alerts.tsx` - SMS subscription feature

---

### 7. Input Validation - POST /validate
**Lambda Function**: `validator-lambda`

**Request**:
```json
{
  "crop": "Tomato",
  "mandi": "Kolar",
  "phoneNumber": "+919876543210"
}
```

**Response**:
```json
{
  "valid": true,
  "errors": []
}
```

**Frontend Usage**: Optional validation before API calls

## Error Handling

### API Service (`src/services/api.ts`)
- **Timeout**: 30 seconds
- **Interceptors**: Automatic error handling and logging
- **Fallback**: Mock data on API failure for development

### Error Response Format
```json
{
  "error": true,
  "message": "Error description",
  "code": "ERROR_CODE"
}
```

### Frontend Error Handling
All components include:
- Try-catch blocks for API calls
- Toast notifications for user feedback
- Fallback to mock data on failure
- Loading states during API calls

## Component Integration

### Dashboard.tsx
**APIs Used**:
- `predictPrice()` - Price prediction
- `getForecast()` - Historical and forecast data
- `getSupplyDemand()` - Supply/demand charts

**Features**:
- AI price prediction with confidence score
- Historical vs forecast price visualization
- Supply vs demand bar charts
- Trend indicators and recommendations

---

### MarketInsights.tsx
**APIs Used**:
- `getMarketInsights()` - Multi-mandi data

**Features**:
- Sortable price comparison table
- Trend indicators (up/down/stable)
- Action recommendations (Buy/Sell/Monitor)
- Highest/lowest/average price cards

---

### RiskSimulation.tsx
**APIs Used**:
- `simulateRisk()` - Risk assessment

**Features**:
- Interactive parameter inputs
- Price range predictions
- Risk level assessment (low/medium/high)
- Scenario analysis with recommendations

---

### Alerts.tsx
**APIs Used**:
- `subscribeToAlerts()` - SMS subscription

**Features**:
- Phone number validation
- Crop and mandi selection
- SMS subscription confirmation
- Alert feature descriptions

## Data Flow

```
User Action → Component State Update → API Call → Loading State
    ↓
API Response → Data Processing → State Update → UI Render
    ↓
Error? → Toast Notification → Fallback to Mock Data
```

## Testing

### Manual Testing
1. Start the development server:
   ```bash
   cd frontend/dashboard
   npm run dev
   ```

2. Test each feature:
   - Dashboard: Select crop/mandi and click "Predict Price"
   - Market Insights: Change crop selection
   - Risk Simulation: Enter parameters and click "Run Simulation"
   - Alerts: Enter phone number and subscribe

### API Testing
Use the browser's Network tab to verify:
- Request payloads
- Response data
- Error handling
- Loading states

## Deployment

### Frontend Deployment
```bash
cd frontend/dashboard
npm run build
# Deploy dist/ folder to hosting service
```

### Environment Variables
Ensure production environment has:
- `VITE_API_URL` set to AWS API Gateway URL
- `VITE_ENV=production`

## Security Considerations

1. **API Keys**: Not exposed in frontend code
2. **CORS**: Configured on API Gateway
3. **Rate Limiting**: Implemented on Lambda functions
4. **Input Validation**: Both frontend and backend validation
5. **Phone Numbers**: Validated before SMS subscription

## Troubleshooting

### Common Issues

**Issue**: API calls failing
- **Solution**: Check `.env` file exists and has correct API URL
- **Solution**: Verify AWS Lambda functions are deployed
- **Solution**: Check browser console for CORS errors

**Issue**: Mock data showing instead of real data
- **Solution**: API error occurred, check Network tab
- **Solution**: Verify Lambda functions are running
- **Solution**: Check API Gateway configuration

**Issue**: TypeScript errors
- **Solution**: Run `npm install` to install dependencies
- **Solution**: Check `vite-env.d.ts` for type definitions

## Future Enhancements

1. **Caching**: Implement Redis for frequently accessed data
2. **WebSockets**: Real-time price updates
3. **Authentication**: User login and personalized alerts
4. **Analytics**: Track API usage and performance
5. **Offline Mode**: Service worker for offline functionality

## Support

For issues or questions:
- Check AWS CloudWatch logs for Lambda errors
- Review API Gateway logs
- Check frontend console for client-side errors
- Verify environment variables are set correctly

---

**Last Updated**: March 9, 2026
**Version**: 1.0.0
**Author**: Elson
