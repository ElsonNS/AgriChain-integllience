import axios from 'axios';
import type { PredictionResult, ForecastData, MarketData, SimulationResult } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://1nz6i4er02.execute-api.us-east-1.amazonaws.com';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data.message || 'Server error occurred');
    } else if (error.request) {
      // Request made but no response
      throw new Error('No response from server. Please check your connection.');
    } else {
      // Error in request setup
      throw new Error('Request failed. Please try again.');
    }
  }
);

// AI Price Prediction - POST /predict
export const predictPrice = async (crop: string, mandi: string): Promise<PredictionResult> => {
  try {
    const response = await api.post('/predict', {
      crop,
      mandi,
      timestamp: new Date().toISOString()
    });
    return response.data;
  } catch (error) {
    console.error('Predict price error:', error);
    throw error;
  }
};

// Price Forecast - POST /predict (using forecast-lambda)
export const getForecast = async (crop: string, mandi: string): Promise<ForecastData[]> => {
  try {
    const response = await api.post('/predict', {
      crop,
      mandi,
      forecast: true,
      days: 14
    });
    return response.data.forecast || [];
  } catch (error) {
    console.error('Get forecast error:', error);
    throw error;
  }
};

// Supply/Demand Data - POST /supply
export const getSupplyDemand = async (crop: string, mandi: string) => {
  try {
    const response = await api.post('/supply', {
      crop,
      mandi
    });
    return response.data;
  } catch (error) {
    console.error('Get supply/demand error:', error);
    throw error;
  }
};

// Market Insights - POST /supply (for multiple mandis)
export const getMarketInsights = async (crop: string): Promise<MarketData[]> => {
  try {
    const response = await api.post('/supply', {
      crop,
      allMandis: true
    });
    return response.data.markets || [];
  } catch (error) {
    console.error('Get market insights error:', error);
    throw error;
  }
};

// Risk Simulation - GET /simulate
export const simulateRisk = async (
  crop: string,
  currentPrice: number,
  rainfall: string,
  transportCost: number
): Promise<SimulationResult> => {
  try {
    const response = await api.get('/simulate', {
      params: {
        crop,
        currentPrice,
        rainfall,
        transportCost,
      }
    });
    return response.data;
  } catch (error) {
    console.error('Simulate risk error:', error);
    throw error;
  }
};

// SMS Alert Subscription - GET /notify
export const subscribeToAlerts = async (
  phoneNumber: string,
  crop: string,
  mandi: string
): Promise<{ success: boolean; message: string }> => {
  try {
    const response = await api.get('/notify', {
      params: {
        phoneNumber,
        crop,
        mandi,
      }
    });
    return {
      success: true,
      message: response.data.message || 'Successfully subscribed to alerts'
    };
  } catch (error) {
    console.error('Subscribe to alerts error:', error);
    throw error;
  }
};

// Validate Input - POST /validate
export const validateInput = async (data: any): Promise<{ valid: boolean; errors?: string[] }> => {
  try {
    const response = await api.post('/validate', data);
    return response.data;
  } catch (error) {
    console.error('Validate input error:', error);
    throw error;
  }
};

export const getMockPrediction = (crop: string, mandi: string): PredictionResult => {
  const basePrice = Math.random() * 50 + 20;
  const trends: Array<'up' | 'down' | 'stable'> = ['up', 'down', 'stable'];
  const trend = trends[Math.floor(Math.random() * trends.length)];

  return {
    predictedPrice: Math.round(basePrice * 100) / 100,
    confidence: Math.round((Math.random() * 20 + 75) * 100) / 100,
    trend,
    recommendation: trend === 'up'
      ? 'Consider holding for better prices'
      : trend === 'down'
      ? 'Sell soon to avoid losses'
      : 'Stable market - good time to sell',
  };
};

export const getMockForecast = (): ForecastData[] => {
  const data: ForecastData[] = [];
  const basePrice = Math.random() * 30 + 25;

  for (let i = 0; i < 14; i++) {
    const date = new Date();
    date.setDate(date.getDate() - 7 + i);
    const variation = (Math.random() - 0.5) * 10;
    data.push({
      date: date.toISOString().split('T')[0],
      price: Math.round((basePrice + variation) * 100) / 100,
    });
  }

  return data;
};

export const getMockMarketData = (): MarketData[] => {
  const mandis = ['Kolar', 'Bangalore', 'Mysore', 'Hubli', 'Belgaum'];
  const trends: Array<'up' | 'down' | 'stable'> = ['up', 'down', 'stable'];

  return mandis.map(mandi => ({
    mandi,
    price: Math.round((Math.random() * 30 + 20) * 100) / 100,
    trend: trends[Math.floor(Math.random() * trends.length)],
    action: 'Monitor',
  }));
};

export const getMockSimulation = (): SimulationResult => {
  const basePrice = Math.random() * 30 + 20;
  const riskLevels: Array<'low' | 'medium' | 'high'> = ['low', 'medium', 'high'];
  const risk = riskLevels[Math.floor(Math.random() * riskLevels.length)];

  return {
    priceRange: {
      min: Math.round((basePrice - 5) * 100) / 100,
      max: Math.round((basePrice + 5) * 100) / 100,
    },
    riskLevel: risk,
    recommendation: risk === 'low'
      ? 'Favorable conditions for selling'
      : risk === 'medium'
      ? 'Monitor market conditions closely'
      : 'High risk - consider alternative strategies',
  };
};
