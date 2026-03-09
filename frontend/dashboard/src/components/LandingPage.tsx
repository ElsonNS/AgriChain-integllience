import { motion } from 'framer-motion';
import { TrendingUp, BarChart3, AlertCircle, MessageSquare, Leaf, ArrowRight } from 'lucide-react';
import { Button } from './ui/Button';

interface LandingPageProps {
  onEnterDashboard: () => void;
}

const AnimatedBackground = () => {
  return (
    <div className="absolute inset-0 overflow-hidden opacity-10">
      <motion.div
        className="absolute top-20 left-20 w-64 h-64 rounded-full bg-[#1F7A4D]"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.3, 0.5, 0.3],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
      <motion.div
        className="absolute bottom-20 right-20 w-96 h-96 rounded-full bg-[#FF9933]"
        animate={{
          scale: [1, 1.3, 1],
          opacity: [0.2, 0.4, 0.2],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: 'easeInOut',
          delay: 1,
        }}
      />
      <motion.div
        className="absolute top-1/2 left-1/2 w-72 h-72 rounded-full bg-[#8D6E63]"
        animate={{
          scale: [1, 1.4, 1],
          opacity: [0.2, 0.3, 0.2],
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: 'easeInOut',
          delay: 2,
        }}
      />
    </div>
  );
};

export const LandingPage = ({ onEnterDashboard }: LandingPageProps) => {
  const features = [
    {
      icon: TrendingUp,
      title: 'AI Price Prediction',
      description: 'Get accurate crop price forecasts powered by advanced machine learning algorithms',
      color: '#1F7A4D',
    },
    {
      icon: BarChart3,
      title: 'Market Trend Analysis',
      description: 'Analyze historical data and market trends across multiple mandis in real-time',
      color: '#FF9933',
    },
    {
      icon: AlertCircle,
      title: 'Risk Simulation Engine',
      description: 'Simulate different scenarios to understand market risks and opportunities',
      color: '#8D6E63',
    },
    {
      icon: MessageSquare,
      title: 'Smart SMS Alerts',
      description: 'Receive timely price alerts and market updates directly on your phone',
      color: '#10b981',
    },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6 },
    },
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#F7F9F8] via-white to-[#F7F9F8] overflow-hidden">
      <AnimatedBackground />

      <nav className="relative z-10 px-6 py-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-2"
          >
            <Leaf className="w-8 h-8 text-[#1F7A4D]" />
            <span className="text-2xl font-bold text-gray-900">AgriPulse AI</span>
          </motion.div>
        </div>
      </nav>

      <motion.main
        className="relative z-10 max-w-7xl mx-auto px-6 pt-20 pb-32"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <div className="text-center mb-20">
          <motion.div variants={itemVariants} className="mb-6">
            <span className="inline-block px-4 py-2 bg-[#1F7A4D]/10 text-[#1F7A4D] rounded-full text-sm font-semibold mb-8">
              AI-Powered Market Intelligence
            </span>
          </motion.div>

          <motion.h1
            variants={itemVariants}
            className="text-5xl md:text-7xl font-bold text-gray-900 mb-8 leading-tight"
          >
            AgriPulse AI
            <br />
            <span className="text-[#1F7A4D]">Predicting Crop Markets</span>
            <br />
            <span className="text-[#FF9933]">with Artificial Intelligence</span>
          </motion.h1>

          <motion.p
            variants={itemVariants}
            className="text-xl text-gray-600 max-w-3xl mx-auto mb-12 leading-relaxed"
          >
            Empower farmers and traders with AI-driven insights to make smarter selling decisions.
            Predict prices, analyze trends, and maximize profits across Indian agricultural markets.
          </motion.p>

          <motion.div variants={itemVariants}>
            <Button onClick={onEnterDashboard} size="lg" className="group">
              Launch Dashboard
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
          </motion.div>
        </div>

        <motion.div
          variants={containerVariants}
          className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mt-20"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              whileHover={{ y: -8 }}
              className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-lg border border-gray-100 hover:shadow-2xl transition-all duration-300"
            >
              <div
                className="w-14 h-14 rounded-xl flex items-center justify-center mb-6"
                style={{ backgroundColor: `${feature.color}15` }}
              >
                <feature.icon className="w-7 h-7" style={{ color: feature.color }} />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
              <p className="text-gray-600 leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          variants={itemVariants}
          className="mt-32 text-center"
        >
          <div className="inline-block p-8 bg-gradient-to-br from-[#1F7A4D]/10 to-[#FF9933]/10 rounded-3xl border border-[#1F7A4D]/20">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Trusted by Thousands of Farmers Across India
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Join the agricultural revolution and make data-driven decisions for your crops
            </p>
          </div>
        </motion.div>
      </motion.main>

      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-white to-transparent pointer-events-none" />
    </div>
  );
};
