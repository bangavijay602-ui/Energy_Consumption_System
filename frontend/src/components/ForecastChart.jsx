import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { LineChart as ChartIcon, Eye, Zap } from 'lucide-react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function ForecastChart({ historyData, forecastData, selectedModel }) {
  if (!forecastData || forecastData.length === 0) return null;

  // Prepare recent history (last 72 observations if available)
  const recentHistory = useMemo(() => {
    if (!historyData || historyData.length === 0) return [];
    const len = historyData.length;
    return historyData.slice(Math.max(0, len - 72));
  }, [historyData]);

  // Merge labels
  const historyLabels = recentHistory.map((d) => d.timestamp || d.DateTime || d.date || d.time || `t-${recentHistory.length - d.index}`);
  const forecastLabels = forecastData.map((d) => d.timestamp);
  const allLabels = [...historyLabels, ...forecastLabels];

  // History dataset: filled for history, null for forecast
  const historyValues = [
    ...recentHistory.map((d) => d.consumption ?? d.Energy_Usage ?? d.load ?? d.power_consumption ?? d.electricity_demand ?? d.value),
    ...Array(forecastData.length).fill(null),
  ];

  // Forecast dataset: connect last history point for continuous visual line
  const lastHistoryVal = historyValues[recentHistory.length - 1];
  const forecastValues = [
    ...Array(Math.max(0, recentHistory.length - 1)).fill(null),
    lastHistoryVal,
    ...forecastData.map((d) => d.predicted_consumption ?? d.forecast_consumption),
  ];

  const data = {
    labels: allLabels.map((ts) => {
      if (typeof ts === 'string' && ts.includes('T')) {
        const parts = ts.split('T');
        return `${parts[0].slice(5)} ${parts[1]?.slice(0, 5)}`;
      }
      if (typeof ts === 'string' && ts.includes(' ')) {
        const parts = ts.split(' ');
        return `${parts[0].slice(5)} ${parts[1]?.slice(0, 5)}`;
      }
      return String(ts);
    }),
    datasets: [
      {
        label: 'Recent Historical Load',
        data: historyValues,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.08)',
        borderWidth: 2.2,
        pointRadius: 0,
        pointHoverRadius: 5,
        fill: true,
        tension: 0.3,
      },
      {
        label: `Genuine Future Forecast (${selectedModel})`,
        data: forecastValues,
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.12)',
        borderWidth: 2.8,
        borderDash: [5, 5],
        pointRadius: 4,
        pointBackgroundColor: '#fbbf24',
        pointBorderColor: '#0a0f1d',
        pointBorderWidth: 1.5,
        pointHoverRadius: 7,
        fill: true,
        tension: 0.3,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#cbd5e1',
          font: { family: 'Inter', size: 12, weight: '500' },
          usePointStyle: true,
          boxWidth: 8,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(10, 15, 29, 0.92)',
        titleColor: '#f8fafc',
        bodyColor: '#cbd5e1',
        borderColor: 'rgba(255, 255, 255, 0.15)',
        borderWidth: 1,
        padding: 12,
        boxPadding: 6,
        usePointStyle: true,
        callbacks: {
          label: (context) => {
            const val = context.raw;
            if (val === null || val === undefined) return null;
            return ` ${context.dataset.label}: ${Number(val).toFixed(2)} kWh`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#64748b', maxTicksLimit: 14, font: { size: 10 } },
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#94a3b8', font: { size: 11 } },
        title: {
          display: true,
          text: 'Energy Consumption (Units)',
          color: '#64748b',
          font: { size: 11, weight: '600' },
        },
      },
    },
  };

  return (
    <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ChartIcon size={20} color="#f59e0b" />
            2. Multi-Step Recursive Future Forecast
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            Dynamic step-by-step lag & rolling feature reconstruction — zero look-ahead data leakage.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <span className="badge badge-emerald">Recent History ({recentHistory.length} pts)</span>
          <span className="badge badge-amber">Forecast ({forecastData.length} pts)</span>
        </div>
      </div>

      <div style={{ height: '380px', position: 'relative' }}>
        <Line data={data} options={options} />
      </div>
    </div>
  );
}
