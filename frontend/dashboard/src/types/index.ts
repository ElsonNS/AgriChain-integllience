export interface PredictionResult {
  predictedPrice: number;
  confidence: number;
  trend: 'up' | 'down' | 'stable';
  recommendation: string;
}

export interface ForecastData {
  date: string;
  price: number;
}

export interface MarketData {
  mandi: string;
  price: number;
  trend: 'up' | 'down' | 'stable';
  action: string;
}

export interface SimulationResult {
  priceRange: {
    min: number;
    max: number;
  };
  riskLevel: 'low' | 'medium' | 'high';
  recommendation: string;
}

export interface ChartDataPoint {
  date: string;
  historical?: number;
  forecast?: number;
  supply?: number;
  demand?: number;
}
