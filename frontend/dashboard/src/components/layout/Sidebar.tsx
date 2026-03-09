import { motion } from 'framer-motion';
import { LayoutDashboard, TrendingUp, AlertCircle, Bell, Leaf, Menu, X } from 'lucide-react';
import { useState } from 'react';

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export const Sidebar = ({ activeTab, onTabChange }: SidebarProps) => {
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'insights', label: 'Market Insights', icon: TrendingUp },
    { id: 'simulation', label: 'Risk Simulation', icon: AlertCircle },
    { id: 'alerts', label: 'Alerts', icon: Bell },
  ];

  const SidebarContent = () => (
    <>
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-[#1F7A4D] to-[#176239] rounded-lg flex items-center justify-center">
            <Leaf className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="font-bold text-gray-900">AgriPulse AI</h2>
            <p className="text-xs text-gray-500">Market Intelligence</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-4">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;

          return (
            <motion.button
              key={item.id}
              onClick={() => {
                onTabChange(item.id);
                setIsMobileOpen(false);
              }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-all duration-200 ${
                isActive
                  ? 'bg-[#1F7A4D] text-white shadow-md'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
              whileHover={!isActive ? { x: 4 } : undefined}
              whileTap={{ scale: 0.98 }}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </motion.button>
          );
        })}
      </nav>

      <div className="p-6 border-t border-gray-100">
        <div className="bg-gradient-to-br from-[#1F7A4D]/10 to-[#FF9933]/10 rounded-lg p-4 border border-[#1F7A4D]/20">
          <p className="text-sm font-semibold text-gray-900 mb-2">AI-Powered Insights</p>
          <p className="text-xs text-gray-600">
            Get real-time market predictions and alerts
          </p>
        </div>
      </div>
    </>
  );

  return (
    <>
      <button
        onClick={() => setIsMobileOpen(!isMobileOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow-lg"
      >
        {isMobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      <motion.aside
        initial={{ x: 0 }}
        className="hidden lg:flex flex-col w-64 bg-white border-r border-gray-100 h-screen fixed left-0 top-0"
      >
        <SidebarContent />
      </motion.aside>

      {isMobileOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setIsMobileOpen(false)}
        >
          <motion.aside
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            className="flex flex-col w-64 bg-white h-screen"
            onClick={(e) => e.stopPropagation()}
          >
            <SidebarContent />
          </motion.aside>
        </motion.div>
      )}
    </>
  );
};
