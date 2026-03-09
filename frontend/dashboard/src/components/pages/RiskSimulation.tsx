import { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Select } from '../ui/Select';
import { Input } from '../ui/Input';
import { CardSkeleton } from '../ui/LoadingSkeleton';
import { Toast } from '../ui/Toast';
import { CROPS, RAINFALL_SCENARIOS } from '../../constants';
import { simulateRisk, getMockSimulation } from '../../services/api';
import type { SimulationResult } from '../../types';

export const RiskSimulation = () => {
  const [crop, setCrop] = useState(CROPS[0]);
  const [currentPrice, setCurrentPrice] = useState('30');
  const [rainfall, setRainfall] = useState(RAINFALL_SCENARIOS[0]);
  const [transportCost, setTransportCost] = useState('5');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showToast, setShowToast] = useState(false);

  const handleSimulate = async () => {
    setLoading(true);
    setError(null);

    try {
      const simulation = await simulateRisk(
        crop,
        parseFloat(currentPrice),
        rainfall,
        parseFloat(transportCost)
      );
      setResult(simulation);
    } catch (err) {
      console.error('Simulation error:', err);
      setError(err instanceof Error ? err.message : 'Failed to run simulation');
      setShowToast(true);
      
      // Fallback to mock data
      const simulation = getMockSimulation();
      setResult(simulation);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk: string) => {
    if (risk === 'low') return 'text-green-600';
    if (risk === 'medium') return 'text-yellow-600';
    return 'text-red-600';
  };

  const getRiskBg = (risk: string) => {
    if (risk === 'low') return 'from-green-50 to-green-100 border-green-200';
    if (risk === 'medium') return 'from-yellow-50 to-yellow-100 border-yellow-200';
    return 'from-red-50 to-red-100 border-red-200';
  };

  const getRiskIcon = (risk: string) => {
    if (risk === 'low') return <CheckCircle className="w-8 h-8 text-green-600" />;
    if (risk === 'medium') return <AlertCircle className="w-8 h-8 text-yellow-600" />;
    return <AlertTriangle className="w-8 h-8 text-red-600" />;
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

      <div>
        <h1 className="text-3xl font-bold text-gray-900">Risk Simulation</h1>
        <p className="text-gray-600 mt-2">Simulate market scenarios and assess risks</p>
      </div>

      <Card className="bg-gradient-to-br from-white to-[#FF9933]/5">
        <h2 className="text-xl font-bold text-gray-900 mb-6">Simulation Parameters</h2>
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <Select
            label="Crop"
            options={CROPS}
            value={crop}
            onChange={setCrop}
          />
          <Input
            label="Current Price (₹/quintal)"
            type="number"
            value={currentPrice}
            onChange={setCurrentPrice}
            placeholder="Enter current price"
          />
          <Select
            label="Rainfall Scenario"
            options={RAINFALL_SCENARIOS}
            value={rainfall}
            onChange={setRainfall}
          />
          <Input
            label="Transport Cost (₹/quintal)"
            type="number"
            value={transportCost}
            onChange={setTransportCost}
            placeholder="Enter transport cost"
          />
        </div>
        <Button
          onClick={handleSimulate}
          loading={loading}
          size="lg"
          variant="secondary"
          className="w-full md:w-auto"
        >
          Run Simulation
        </Button>
      </Card>

      {loading && (
        <div className="grid md:grid-cols-3 gap-6">
          {[...Array(3)].map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      )}

      {result && !loading && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-6"
        >
          <div className="grid md:grid-cols-3 gap-6">
            <Card className="bg-gradient-to-br from-[#1F7A4D] to-[#176239] text-white">
              <p className="text-sm opacity-90 mb-2">Minimum Price</p>
              <p className="text-4xl font-bold">₹{result.priceRange.min}</p>
              <p className="text-sm opacity-75 mt-1">per quintal</p>
            </Card>

            <Card className="bg-gradient-to-br from-[#FF9933] to-[#e68a2e] text-white">
              <p className="text-sm opacity-90 mb-2">Maximum Price</p>
              <p className="text-4xl font-bold">₹{result.priceRange.max}</p>
              <p className="text-sm opacity-75 mt-1">per quintal</p>
            </Card>

            <Card className="bg-gradient-to-br from-[#8D6E63] to-[#6d534e] text-white">
              <p className="text-sm opacity-90 mb-2">Price Range</p>
              <p className="text-4xl font-bold">
                ₹{(result.priceRange.max - result.priceRange.min).toFixed(2)}
              </p>
              <p className="text-sm opacity-75 mt-1">variance</p>
            </Card>
          </div>

          <Card className={`bg-gradient-to-br ${getRiskBg(result.riskLevel)} border-2`} hover={false}>
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0">
                {getRiskIcon(result.riskLevel)}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-3">
                  <h3 className="text-lg font-bold text-gray-900">Risk Assessment</h3>
                  <span className={`text-2xl font-bold ${getRiskColor(result.riskLevel)} uppercase`}>
                    {result.riskLevel}
                  </span>
                </div>
                <p className="text-gray-700 font-medium">{result.recommendation}</p>
              </div>
            </div>
          </Card>

          <Card>
            <h3 className="text-lg font-bold text-gray-900 mb-4">Scenario Analysis</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-semibold text-gray-900">Current Crop</p>
                  <p className="text-sm text-gray-600">{crop}</p>
                </div>
                <div className="text-right">
                  <p className="font-semibold text-gray-900">₹{currentPrice}</p>
                  <p className="text-sm text-gray-600">Current Price</p>
                </div>
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-semibold text-gray-900">Rainfall Condition</p>
                  <p className="text-sm text-gray-600">{rainfall}</p>
                </div>
                <div className="text-right">
                  <p className="font-semibold text-gray-900">₹{transportCost}</p>
                  <p className="text-sm text-gray-600">Transport Cost</p>
                </div>
              </div>

              <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                <p className="text-sm font-semibold text-gray-900 mb-2">Key Insights</p>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Price volatility expected due to {rainfall.toLowerCase()} rainfall conditions</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Transport costs may impact final margins by ₹{transportCost}/quintal</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Market demand for {crop} shows {result.riskLevel} risk patterns</span>
                  </li>
                </ul>
              </div>
            </div>
          </Card>
        </motion.div>
      )}

      {!result && !loading && (
        <Card className="text-center py-16">
          <div className="max-w-md mx-auto">
            <div className="w-20 h-20 bg-[#FF9933]/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <AlertCircle className="w-10 h-10 text-[#FF9933]" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">Ready to Simulate</h3>
            <p className="text-gray-600">
              Enter simulation parameters and click "Run Simulation" to see risk analysis
            </p>
          </div>
        </Card>
      )}
    </div>
  );
};
