import { motion } from 'framer-motion';

interface SelectProps {
  label?: string;
  options: string[];
  value: string;
  onChange: (value: string) => void;
  className?: string;
}

export const Select = ({ label, options, value, onChange, className = '' }: SelectProps) => {
  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      {label && (
        <label className="text-sm font-medium text-gray-700">
          {label}
        </label>
      )}
      <motion.select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="px-4 py-3 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#1F7A4D] focus:border-transparent transition-all duration-200 bg-white cursor-pointer"
        whileFocus={{ scale: 1.01 }}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </motion.select>
    </div>
  );
};
