import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, ArrowUpDown } from 'lucide-react';
import { Card } from '../ui/Card';
import { Select } from '../ui/Select';
import { Toast } from '../ui/Toast';
import { CardSkeleton } from '../ui/LoadingSkeleton';
import { getMarketInsights, getMockMarketData } from '../../services/api';
import { CROPS } from '../../constants';
import type { MarketData } from '../../types';

export const MarketInsights = () => {
  const [crop, setCrop] = useState(CROPS[0]);
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [sortField, setSortField] = useState<keyof MarketData>('price');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showToast, setShowToast] = useState(false);

  useEffect(() => {
    const fetchMarketData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const data = await getMarketInsights(crop);
        setMarketData(data);
      } catch (err) {
        console.error('Market insights error:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch market data');
        setShowToast(true);
        
        // Fallback to mock data
        const mockData = getMockMarketData();
        setMarketData(mockData);
      } finally {
        setLoading(false);
      }
    };

    fetchMarketData();
  }, [crop]);

  const handleSort = (field: keyof MarketData) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const sortedData = [...marketData].sort((a, b) => {
    if (sortField === 'price') {
      return sortDirection === 'asc' ? a.price - b.price : b.price - a.price;
    }
    return 0;
  });

  const getTrendIcon = (trend: string) => {
    if (trend === 'up') return <TrendingUp className="w-4 h-4 text-green-500" />;
    if (trend === 'down') return <TrendingDown className="w-4 h-4 text-red-500" />;
    return <Minus className="w-4 h-4 text-gray-500" />;
  };

  const getTrendBadge = (trend: string) => {
    const colors = {
      up: 'bg-green-50 text-green-700 border-green-200',
      down: 'bg-red-50 text-red-700 border-red-200',
      stable: 'bg-gray-50 text-gray-700 border-gray-200',
    };
    return colors[trend as keyof typeof colors];
  };

  const getActionBadge = (action: string) => {
    if (action === 'Buy') return 'bg-blue-50 text-blue-700 border-blue-200';
    if (action === 'Sell') return 'bg-orange-50 text-orange-700 border-orange-200';
    return 'bg-purple-50 text-purple-700 border-purple-200';
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
        <h1 className="text-3xl font-bold text-gray-900">Market Insights</h1>
        <p className="text-gray-600 mt-2">Compare prices across different mandis</p>
      </div>

      {loading ? (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      ) : (
        <>
      <Card>
        <div className="mb-6">
          <Select
            label="Select Crop"
            options={CROPS}
            value={crop}
            onChange={setCrop}
            className="max-w-xs"
          />
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-4 px-4 font-semibold text-gray-700">
                  Mandi
                </th>
                <th
                  className="text-left py-4 px-4 font-semibold text-gray-700 cursor-pointer hover:bg-gray-50 transition-colors"
                  onClick={() => handleSort('price')}
                >
                  <div className="flex items-center gap-2">
                    Price (₹/quintal)
                    <ArrowUpDown className="w-4 h-4" />
                  </div>
                </th>
                <th className="text-left py-4 px-4 font-semibold text-gray-700">
                  Trend
                </th>
                <th className="text-left py-4 px-4 font-semibold text-gray-700">
                  Best Action
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedData.map((item, index) => (
                <motion.tr
                  key={item.mandi}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="border-b border-gray-100 hover:bg-gray-50 transition-colors"
                >
                  <td className="py-4 px-4">
                    <span className="font-semibold text-gray-900">{item.mandi}</span>
                  </td>
                  <td className="py-4 px-4">
                    <span className="text-2xl font-bold text-[#1F7A4D]">
                      ₹{item.price}
                    </span>
                  </td>
                  <td className="py-4 px-4">
                    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border ${getTrendBadge(item.trend)}`}>
                      {getTrendIcon(item.trend)}
                      <span className="text-sm font-medium capitalize">{item.trend}</span>
                    </div>
                  </td>
                  <td className="py-4 px-4">
                    <span className={`inline-block px-3 py-1 rounded-full border text-sm font-medium ${getActionBadge(item.action)}`}>
                      {item.action}
                    </span>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid md:grid-cols-3 gap-6">
        <Card className="bg-gradient-to-br from-[#1F7A4D] to-[#176239] text-white">
          <p className="text-sm opacity-90 mb-2">Highest Price</p>
          <p className="text-3xl font-bold">
            ₹{Math.max(...marketData.map(d => d.price)).toFixed(2)}
          </p>
          <p className="text-sm opacity-75 mt-1">
            {marketData.find(d => d.price === Math.max(...marketData.map(m => m.price)))?.mandi}
          </p>
        </Card>

        <Card className="bg-gradient-to-br from-[#FF9933] to-[#e68a2e] text-white">
          <p className="text-sm opacity-90 mb-2">Lowest Price</p>
          <p className="text-3xl font-bold">
            ₹{Math.min(...marketData.map(d => d.price)).toFixed(2)}
          </p>
          <p className="text-sm opacity-75 mt-1">
            {marketData.find(d => d.price === Math.min(...marketData.map(m => m.price)))?.mandi}
          </p>
        </Card>

        <Card className="bg-gradient-to-br from-[#8D6E63] to-[#6d534e] text-white">
          <p className="text-sm opacity-90 mb-2">Average Price</p>
          <p className="text-3xl font-bold">
            ₹{(marketData.reduce((sum, d) => sum + d.price, 0) / marketData.length).toFixed(2)}
          </p>
          <p className="text-sm opacity-75 mt-1">Across all mandis</p>
        </Card>
      </div>
        </>
      )}
    </div>
  );
};
