import { useState } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, ArrowRight } from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Select } from '../ui/Select';
import { CardSkeleton } from '../ui/LoadingSkeleton';
import { Toast } from '../ui/Toast';
import { CROPS, MANDIS } from '../../constants';
import { predictPrice, getForecast, getSupplyDemand, getMockPrediction, getMockForecast } from '../../services/api';
import type { PredictionResult, ChartDataPoint } from '../../types';

export const Dashboard = () => {
  const [crop, setCrop] = useState(CROPS[0]);
  const [mandi, setMandi] = useState(MANDIS[0]);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showToast, setShowToast] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);

    try {
      // Call real API endpoints
      const [predictionResult, forecastData, supplyDemandData] = await Promise.all([
        predictPrice(crop, mandi),
        getForecast(crop, mandi),
        getSupplyDemand(crop, mandi)
      ]);

      setPrediction(predictionResult);

      // Format chart data combining historical and forecast
      const formattedData: ChartDataPoint[] = forecastData.map((item, idx) => ({
        date: item.date,
        historical: idx < 7 ? item.price : undefined,
        forecast: idx >= 6 ? item.price : undefined,
        supply: supplyDemandData?.supply?.[idx] || 50 + Math.random() * 30,
        demand: supplyDemandData?.demand?.[idx] || 40 + Math.random() * 40,
      }));

      setChartData(formattedData);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch prediction. Please try again.');
      setShowToast(true);
      
      // Fallback to mock data on error
      const result = getMockPrediction(crop, mandi);
      setPrediction(result);
      const forecast = getMockForecast();
      const formattedData: ChartDataPoint[] = forecast.map((item, idx) => ({
        date: item.date,
        historical: idx < 7 ? item.price : undefined,
        forecast: idx >= 6 ? item.price : undefined,
        supply: 50 + Math.random() * 30,
        demand: 40 + Math.random() * 40,
      }));
      setChartData(formattedData);
    } finally {
      setLoading(false);
    }
  };

  const getTrendIcon = (trend: string) => {
    if (trend === 'up') return <TrendingUp className="w-5 h-5 text-green-500" />;
    if (trend === 'down') return <TrendingDown className="w-5 h-5 text-red-500" />;
    return <Minus className="w-5 h-5 text-gray-500" />;
  };

  const getTrendColor = (trend: string) => {
    if (trend === 'up') return 'text-green-600';
    if (trend === 'down') return 'text-red-600';
    return 'text-gray-600';
  };

  return (
    <div className="space-y-6">
      {showToast && error && (
        <Toast
          message={error}
          type="error"
          onClose={() => setShowToast(false)}
        />
      )}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">AI Price Prediction</h1>
          <p className="text-gray-600 mt-2">Predict crop prices with advanced machine learning</p>
        </div>
      </div>

      <Card className="bg-gradient-to-br from-white to-[#1F7A4D]/5">
        <div className="grid md:grid-cols-3 gap-6">
          <Select
            label="Select Crop"
            options={CROPS}
            value={crop}
            onChange={setCrop}
          />
          <Select
            label="Select Mandi"
            options={MANDIS}
            value={mandi}
            onChange={setMandi}
          />
          <div className="flex items-end">
            <Button
              onClick={handlePredict}
              loading={loading}
              className="w-full"
              size="lg"
            >
              Predict Price
              <ArrowRight className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </Card>

      {loading && (
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      )}

      {prediction && !loading && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
            <Card className="bg-gradient-to-br from-[#1F7A4D] to-[#176239] text-white">
              <p className="text-sm opacity-90 mb-2">Predicted Price</p>
              <p className="text-4xl font-bold mb-1">₹{prediction.predictedPrice}</p>
              <p className="text-sm opacity-75">per quintal</p>
            </Card>

            <Card className="bg-gradient-to-br from-[#FF9933] to-[#e68a2e] text-white">
              <p className="text-sm opacity-90 mb-2">Confidence Score</p>
              <p className="text-4xl font-bold mb-1">{prediction.confidence}%</p>
              <div className="w-full bg-white/20 rounded-full h-2 mt-2">
                <div
                  className="bg-white rounded-full h-2 transition-all duration-1000"
                  style={{ width: `${prediction.confidence}%` }}
                />
              </div>
            </Card>

            <Card>
              <p className="text-sm text-gray-600 mb-2">Price Trend</p>
              <div className="flex items-center gap-2 mb-2">
                {getTrendIcon(prediction.trend)}
                <span className={`text-2xl font-bold ${getTrendColor(prediction.trend)}`}>
                  {prediction.trend.charAt(0).toUpperCase() + prediction.trend.slice(1)}
                </span>
              </div>
              <p className="text-xs text-gray-500">Based on AI analysis</p>
            </Card>

            <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
              <p className="text-sm text-gray-700 mb-2">AI Recommendation</p>
              <p className="text-sm font-semibold text-gray-900">
                {prediction.recommendation}
              </p>
            </Card>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            <Card hover={false}>
              <h3 className="text-lg font-bold text-gray-900 mb-4">Historical & Forecast Prices</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorHistorical" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#1F7A4D" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#1F7A4D" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#FF9933" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#FF9933" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="date" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="historical"
                    stroke="#1F7A4D"
                    strokeWidth={3}
                    fillOpacity={1}
                    fill="url(#colorHistorical)"
                  />
                  <Area
                    type="monotone"
                    dataKey="forecast"
                    stroke="#FF9933"
                    strokeWidth={3}
                    strokeDasharray="5 5"
                    fillOpacity={1}
                    fill="url(#colorForecast)"
                  />
                </AreaChart>
              </ResponsiveContainer>
              <div className="flex items-center justify-center gap-6 mt-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-[#1F7A4D] rounded-full" />
                  <span className="text-sm text-gray-600">Historical</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-[#FF9933] rounded-full" />
                  <span className="text-sm text-gray-600">Forecast</span>
                </div>
              </div>
            </Card>

            <Card hover={false}>
              <h3 className="text-lg font-bold text-gray-900 mb-4">Supply vs Demand</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData.slice(-7)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="date" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Bar dataKey="supply" fill="#1F7A4D" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="demand" fill="#FF9933" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="flex items-center justify-center gap-6 mt-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-[#1F7A4D] rounded-full" />
                  <span className="text-sm text-gray-600">Supply</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-[#FF9933] rounded-full" />
                  <span className="text-sm text-gray-600">Demand</span>
                </div>
              </div>
            </Card>
          </div>
        </motion.div>
      )}

      {!prediction && !loading && (
        <Card className="text-center py-16">
          <div className="max-w-md mx-auto">
            <div className="w-20 h-20 bg-[#1F7A4D]/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <TrendingUp className="w-10 h-10 text-[#1F7A4D]" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">Ready to Predict</h3>
            <p className="text-gray-600">
              Select a crop and mandi, then click "Predict Price" to see AI-powered insights
            </p>
          </div>
        </Card>
      )}
    </div>
  );
};
