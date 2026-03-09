# AgriChain – AI-Powered Agricultural Market Intelligence for India

**Created by**: Elson

## 🌾 Overview

AgriChain is a comprehensive agricultural market intelligence platform that combines a modern React frontend with a serverless AWS backend to provide AI-driven insights, market data, and real-time alerts for agricultural commodities in India.

## 🏗️ Architecture

### Frontend
- **Framework**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Location**: `frontend/dashboard/`

### Backend
- **Platform**: AWS Lambda + API Gateway
- **Database**: DynamoDB (3 tables)
- **Architecture**: Serverless
- **Location**: `backend/`

### AI Component
- **Implementation**: Linear regression and statistical models within Lambda functions
- **Purpose**: Price prediction, forecasting, and risk analysis
- **Location**: Embedded in `backend/services/forecast/` Lambda function

## 📁 Project Structure

```
AgriChain-integllience/
├── frontend/
│   └── dashboard/              # React application
│       ├── src/
│       │   ├── components/     # UI components
│       │   │   ├── pages/      # Page components
│       │   │   ├── ui/         # Reusable UI components
│       │   │   └── layout/     # Layout components
│       │   ├── services/       # API integration
│       │   ├── types/          # TypeScript types
│       │   └── constants/      # App constants
│       ├── .env                # Environment variables
│       └── package.json
├── backend/
│   ├── services/
│   │   ├── forecast/           # AI-powered forecast-lambda (linear regression)
│   │   ├── supply/             # supply-lambda
│   │   ├── simulation/         # simulation-lambda (risk analysis)
│   │   └── offer-validator/    # validator-lambda
│   └── common/                 # Shared utilities
├── infrastructure/             # AWS infrastructure code
├── API_INTEGRATION_GUIDE.md    # Detailed API documentation
└── README.md                   # This file
```

## 🚀 Features

### 1. **Landing Page**
- Animated hero section
- Feature highlights
- Modern SaaS-style design

### 2. **AI Price Prediction Dashboard**
- Real-time price predictions with confidence scores
- Historical vs forecast price visualization
- Supply vs demand analysis
- Trend indicators (up/down/stable)
- AI-powered recommendations

### 3. **Market Insights**
- Multi-mandi price comparison
- Sortable data table
- Trend indicators for each market
- Action recommendations (Buy/Sell/Monitor)
- Highest/lowest/average price analytics

### 4. **Risk Simulation**
- Interactive scenario modeling
- Price range predictions
- Risk level assessment (low/medium/high)
- Weather impact analysis
- Transport cost considerations

### 5. **Smart Alerts**
- SMS subscription for price alerts
- Crop and mandi-specific notifications
- Phone number validation
- Real-time market updates

## 🔌 API Endpoints

**Base URL**: `https://1nz6i4er02.execute-api.us-east-1.amazonaws.com`

| Endpoint | Method | Lambda Function | Purpose |
|----------|--------|-----------------|---------|
| `/predict` | POST | forecast-lambda | AI price prediction |
| `/supply` | POST | supply-lambda | Supply/demand data |
| `/simulate` | GET | simulation-lambda | Risk simulation |
| `/notify` | GET | send-sms-alert | SMS alerts |
| `/validate` | POST | validator-lambda | Input validation |

See [API_INTEGRATION_GUIDE.md](./API_INTEGRATION_GUIDE.md) for detailed documentation.

## 🗄️ DynamoDB Tables

1. **mandi_market_data** - Market price data
2. **crops_metadata** - Crop information
3. **price_history** - Historical price records

## 🛠️ Setup & Installation

### Prerequisites
- Node.js 18+ and npm
- AWS Account (for backend deployment)
- Python 3.9+ (for Lambda functions with AI algorithms)

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend/dashboard

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Update .env with your API URL
# VITE_API_URL=https://1nz6i4er02.execute-api.us-east-1.amazonaws.com

# Start development server
npm run dev

# Build for production
npm run build
```

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies for each Lambda function
cd services/forecast && pip install -r requirements.txt
cd ../supply && pip install -r requirements.txt
cd ../simulation && pip install -r requirements.txt
cd ../offer-validator && pip install -r requirements.txt

# Deploy to AWS (using your preferred method)
# - AWS SAM
# - Serverless Framework
# - AWS CDK
# - Manual deployment via AWS Console
```

### AI Implementation

The AI algorithms (linear regression, statistical analysis) are implemented directly within the Lambda functions:
- **forecast-lambda**: Uses linear regression for price prediction
- **simulation-lambda**: Statistical models for risk assessment
- No separate ML model training required - algorithms run in real-time

## 🔧 Configuration

### Environment Variables

Create `frontend/dashboard/.env`:

```env
VITE_API_URL=
VITE_ENV=production
```

### AWS Configuration

Ensure your AWS Lambda functions have:
- Proper IAM roles for DynamoDB access
- API Gateway configured with CORS
- Environment variables for DynamoDB table names
- Appropriate timeout settings (30s recommended)

## 🧪 Testing

### Frontend Testing

```bash
cd frontend/dashboard

# Run development server
npm run dev

# Test each feature:
# 1. Dashboard - Select crop/mandi and predict price
# 2. Market Insights - View multi-mandi comparison
# 3. Risk Simulation - Run scenario analysis
# 4. Alerts - Subscribe to SMS notifications
```

### API Testing

Use browser DevTools Network tab or tools like Postman to test:
- Request/response payloads
- Error handling
- Loading states
- Fallback to mock data

## 📊 Data Flow

```
User Input → React Component → API Service (Axios)
    ↓
AWS API Gateway → Lambda Function → DynamoDB
    ↓
Response → Error Handling → State Update → UI Render
```

## 🎨 UI Components

### Reusable Components
- `Button` - Primary/secondary variants with loading states
- `Card` - Container with hover effects
- `Input` - Form input with validation
- `Select` - Dropdown selector
- `Toast` - Success/error notifications
- `LoadingSkeleton` - Loading placeholders

### Page Components
- `LandingPage` - Hero and features
- `Dashboard` - AI predictions and charts
- `MarketInsights` - Multi-mandi comparison
- `RiskSimulation` - Scenario analysis
- `Alerts` - SMS subscription

## 🔐 Security

- Environment variables for sensitive data
- Input validation on frontend and backend
- CORS configured on API Gateway
- Rate limiting on Lambda functions
- Phone number validation for SMS alerts

## 🚢 Deployment

### Frontend Deployment

```bash
cd frontend/dashboard
npm run build

# Deploy dist/ folder to:
# - Vercel
# - Netlify
# - AWS S3 + CloudFront
# - Any static hosting service
```

### Backend Deployment

Deploy Lambda functions using:
- AWS SAM CLI
- Serverless Framework
- AWS CDK
- Manual deployment via AWS Console

## 📈 Performance

- **Frontend**: Optimized with Vite for fast builds
- **Backend**: Serverless architecture scales automatically
- **Caching**: Consider adding Redis for frequently accessed data
- **CDN**: Use CloudFront for static assets

## 🐛 Troubleshooting

### Common Issues

**API calls failing**
- Check `.env` file exists with correct API URL
- Verify Lambda functions are deployed
- Check CORS configuration on API Gateway

**TypeScript errors**
- Run `npm install` to install dependencies
- Check `vite-env.d.ts` for type definitions

**Mock data showing**
- API error occurred, check browser console
- Verify Lambda functions are running
- Check CloudWatch logs for errors

## 🔮 Future Enhancements

1. **Real-time Updates**: WebSocket integration for live prices
2. **User Authentication**: Login and personalized dashboards
3. **Advanced Analytics**: ML-powered trend analysis
4. **Mobile App**: React Native version
5. **Offline Mode**: Service worker for offline functionality
6. **Multi-language**: Support for regional languages
7. **Export Features**: PDF/Excel report generation
8. **Weather Integration**: Real-time weather data

## 📝 Documentation

- [API Integration Guide](./API_INTEGRATION_GUIDE.md) - Detailed API documentation
- [Testing Guide](./TESTING_GUIDE.md) - Testing procedures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is proprietary software created by Elson.

## 👤 Author

**Elson**
- Project: AgriChain
- Purpose: AI-Powered Agricultural Market Intelligence for India

## 🙏 Acknowledgments

- AWS for serverless infrastructure
- React community for excellent tools
- Indian agricultural market data providers

---

**Version**: 1.0.0  
**Last Updated**: March 9, 2026  
**Status**: Production Ready ✅
