import { useState } from 'react';
import { motion } from 'framer-motion';
import { Bell, Smartphone, Mail, CheckCircle } from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Select } from '../ui/Select';
import { Input } from '../ui/Input';
import { Toast } from '../ui/Toast';
import { CROPS, MANDIS } from '../../constants';
import { subscribeToAlerts } from '../../services/api';

export const Alerts = () => {
  const [phoneNumber, setPhoneNumber] = useState('');
  const [crop, setCrop] = useState(CROPS[0]);
  const [mandi, setMandi] = useState(MANDIS[0]);
  const [loading, setLoading] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [toastType, setToastType] = useState<'success' | 'error'>('success');
  const [error, setError] = useState<string | null>(null);

  const handleSubscribe = async () => {
    // Validate phone number
    if (!phoneNumber || phoneNumber.length < 10) {
      setError('Please enter a valid phone number');
      setToastMessage('Please enter a valid phone number');
      setToastType('error');
      setShowToast(true);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await subscribeToAlerts(phoneNumber, crop, mandi);
      setToastMessage(result.message || 'Successfully subscribed to price alerts!');
      setToastType('success');
      setShowToast(true);
      setPhoneNumber('');
    } catch (err) {
      console.error('Subscribe error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to subscribe. Please try again.';
      setError(errorMessage);
      setToastMessage(errorMessage);
      setToastType('error');
      setShowToast(true);
    } finally {
      setLoading(false);
    }
  };

  const alertTypes = [
    {
      icon: Bell,
      title: 'Price Alerts',
      description: 'Get notified when prices reach your target levels',
      color: '#1F7A4D',
    },
    {
      icon: Smartphone,
      title: 'SMS Updates',
      description: 'Receive daily market updates via SMS',
      color: '#FF9933',
    },
    {
      icon: Mail,
      title: 'Market Reports',
      description: 'Weekly market analysis reports via email',
      color: '#8D6E63',
    },
  ];

  return (
    <div className="space-y-6">
      {showToast && (
        <Toast
          message={toastMessage}
          type={toastType}
          onClose={() => setShowToast(false)}
        />
      )}

      <div>
        <h1 className="text-3xl font-bold text-gray-900">Smart Alerts</h1>
        <p className="text-gray-600 mt-2">Subscribe to real-time market updates</p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {alertTypes.map((alert, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="text-center h-full">
              <div
                className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
                style={{ backgroundColor: `${alert.color}15` }}
              >
                <alert.icon className="w-8 h-8" style={{ color: alert.color }} />
              </div>
              <h3 className="text-lg font-bold text-gray-900 mb-2">{alert.title}</h3>
              <p className="text-gray-600 text-sm">{alert.description}</p>
            </Card>
          </motion.div>
        ))}
      </div>

      <Card className="bg-gradient-to-br from-white to-[#1F7A4D]/5">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-[#1F7A4D] to-[#176239] rounded-full flex items-center justify-center mx-auto mb-4">
              <Bell className="w-8 h-8 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Subscribe to Price Alerts</h2>
            <p className="text-gray-600">
              Enter your details below to receive SMS alerts for your selected crop and mandi
            </p>
          </div>

          <div className="space-y-6">
            <Input
              label="Phone Number"
              type="tel"
              value={phoneNumber}
              onChange={setPhoneNumber}
              placeholder="+91 XXXXX XXXXX"
            />

            <div className="grid md:grid-cols-2 gap-6">
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
            </div>

            <Button
              onClick={handleSubscribe}
              loading={loading}
              size="lg"
              className="w-full"
              disabled={!phoneNumber}
            >
              Subscribe to AI Price Alerts
            </Button>
          </div>
        </div>
      </Card>

      <Card>
        <h3 className="text-lg font-bold text-gray-900 mb-4">Alert Features</h3>
        <div className="space-y-4">
          <div className="flex items-start gap-3 p-4 bg-green-50 rounded-lg border border-green-200">
            <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
            <div>
              <p className="font-semibold text-gray-900">Real-time Price Updates</p>
              <p className="text-sm text-gray-600">
                Receive instant notifications when market prices change significantly
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <CheckCircle className="w-5 h-5 text-blue-600 mt-0.5" />
            <div>
              <p className="font-semibold text-gray-900">AI-Powered Recommendations</p>
              <p className="text-sm text-gray-600">
                Get smart selling suggestions based on market trends and predictions
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3 p-4 bg-purple-50 rounded-lg border border-purple-200">
            <CheckCircle className="w-5 h-5 text-purple-600 mt-0.5" />
            <div>
              <p className="font-semibold text-gray-900">Market Volatility Warnings</p>
              <p className="text-sm text-gray-600">
                Be alerted to sudden market changes and risk factors
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3 p-4 bg-orange-50 rounded-lg border border-orange-200">
            <CheckCircle className="w-5 h-5 text-orange-600 mt-0.5" />
            <div>
              <p className="font-semibold text-gray-900">Weather Impact Alerts</p>
              <p className="text-sm text-gray-600">
                Stay informed about weather conditions affecting crop prices
              </p>
            </div>
          </div>
        </div>
      </Card>

      <Card className="bg-gradient-to-br from-[#FF9933]/10 to-[#1F7A4D]/10 border-[#1F7A4D]/20">
        <div className="text-center">
          <h3 className="text-xl font-bold text-gray-900 mb-2">Stay Ahead of the Market</h3>
          <p className="text-gray-700">
            Join thousands of farmers and traders who trust AgriPulse AI for timely market updates.
            Never miss an opportunity to sell at the best price.
          </p>
        </div>
      </Card>
    </div>
  );
};
