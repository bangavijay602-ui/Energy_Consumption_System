import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import UploadSection from './components/UploadSection';
import PipelineStepper from './components/PipelineStepper';
import DatasetIntelligenceCard from './components/DatasetIntelligenceCard';
import ModelLeaderboard from './components/ModelLeaderboard';
import ForecastChart from './components/ForecastChart';
import ForecastTable from './components/ForecastTable';
import { AlertCircle, CheckCircle2, Zap } from 'lucide-react';

const API_BASE_URL = 'https://energy-consumption-system.onrender.com';

export default function App() {
  const [file, setFile] = useState(null);
  const [rawHistoryRows, setRawHistoryRows] = useState([]);
  const [horizon, setHorizon] = useState(24);
  const [selectedPreset, setSelectedPreset] = useState(null);

  const [apiOnline, setApiOnline] = useState(false);
  const [checkingApi, setCheckingApi] = useState(false);

  const [pipelineStatus, setPipelineStatus] = useState('idle'); // 'idle' | 'running' | 'complete' | 'error'
  const [pipelineStep, setPipelineStep] = useState(1);
  const [errorMsg, setErrorMsg] = useState(null);

  const [pipelineResult, setPipelineResult] = useState(null);

  // Check FastAPI connection on mount
  const checkHealth = async () => {
    setCheckingApi(true);
    try {
      const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
      if (res.ok) {
        setApiOnline(true);
      } else {
        setApiOnline(false);
      }
    } catch {
      setApiOnline(false);
    } finally {
      setCheckingApi(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  // Preset dataset handler
  const handleSelectPreset = async (preset) => {
    setSelectedPreset(preset.id);
    setErrorMsg(null);

    // Generate client-side sample CSV
    let csvText = '';
    const now = new Date(2026, 0, 1, 0, 0, 0);

    if (preset.id === 'dataset1') {
      // Hourly: DateTime, Energy_Usage (720 rows)
      const rows = ['DateTime,Energy_Usage'];
      for (let i = 0; i < 720; i++) {
        const d = new Date(now.getTime() + i * 3600000);
        const hour = d.getHours();
        const load = (15 + 8 * Math.sin((2 * Math.PI * (hour - 6)) / 24) + (Math.random() - 0.5) * 3).toFixed(2);
        rows.push(`${d.toISOString().replace('T', ' ').slice(0, 19)},${load}`);
      }
      csvText = rows.join('\n');
    } else if (preset.id === 'dataset2') {
      // 30-min: timestamp, load (1000 rows)
      const rows = ['timestamp,load'];
      for (let i = 0; i < 1000; i++) {
        const d = new Date(now.getTime() + i * 1800000);
        const hour = d.getHours() + d.getMinutes() / 60;
        let load = 120 + 40 * Math.sin((2 * Math.PI * (hour - 7)) / 24) + (Math.random() - 0.5) * 10;
        if (i === 350) load += 80; // Peak
        rows.push(`${d.toISOString().slice(0, 19)},${load.toFixed(2)}`);
      }
      csvText = rows.join('\n');
    } else if (preset.id === 'dataset3') {
      // 15-min: reading_time, power_consumption, temperature, humidity (1500 rows)
      const rows = ['reading_time,power_consumption,temperature,humidity'];
      for (let i = 0; i < 1500; i++) {
        const d = new Date(now.getTime() + i * 900000);
        const hour = d.getHours() + d.getMinutes() / 60;
        const temp = (18 + 10 * Math.sin((2 * Math.PI * (hour - 9)) / 24) + (Math.random() - 0.5) * 2).toFixed(1);
        const hum = (60 - 20 * Math.sin((2 * Math.PI * (hour - 9)) / 24) + (Math.random() - 0.5) * 5).toFixed(1);
        const power = (45 + 15 * Math.sin((2 * Math.PI * (hour - 6)) / 24) + (Math.random() - 0.5) * 4).toFixed(2);
        rows.push(`${d.toISOString().slice(0, 16).replace('T', ' ')},${power},${temp},${hum}`);
      }
      csvText = rows.join('\n');
    } else if (preset.id === 'dataset4') {
      // Daily: date, electricity_demand (365 rows)
      const rows = ['date,electricity_demand'];
      for (let i = 0; i < 365; i++) {
        const d = new Date(now.getTime() + i * 86400000);
        const dem = (500 + 120 * Math.sin((2 * Math.PI * (i - 15)) / 365) + (Math.random() - 0.5) * 40).toFixed(1);
        rows.push(`${d.toISOString().slice(0, 10)},${dem}`);
      }
      csvText = rows.join('\n');
    } else {
      // Hourly: time, value, temp
      const rows = ['time,value,temp'];
      for (let i = 0; i < 600; i++) {
        const d = new Date(now.getTime() + i * 3600000);
        const hour = d.getHours();
        const temp = (15 + 8 * Math.sin((2 * Math.PI * (hour - 8)) / 24) + (Math.random() - 0.5) * 2).toFixed(1);
        const val = (30 + 12 * Math.sin((2 * Math.PI * (hour - 5)) / 24) + (Math.random() - 0.5) * 3).toFixed(2);
        rows.push(`${d.toISOString().slice(0, 16).replace('T', ' ')},${val},${temp}`);
      }
      csvText = rows.join('\n');
    }

    const blob = new Blob([csvText], { type: 'text/csv' });
    const dummyFile = new File([blob], `${preset.id}.csv`, { type: 'text/csv' });
    setFile(dummyFile);
  };

  // Run training & forecasting pipeline via FastAPI
  const handleRunPipeline = async () => {
    if (!file) return;

    setPipelineStatus('running');
    setPipelineStep(1);
    setErrorMsg(null);
    setPipelineResult(null);

    // Animated progression through steps
    const stepInterval = setInterval(() => {
      setPipelineStep((prev) => (prev < 10 ? prev + 1 : prev));
    }, 450);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('horizon', horizon);

      const response = await fetch(`${API_BASE_URL}/api/forecast`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(stepInterval);

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || errData.message || 'Pipeline execution failed.');
      }

      const data = await response.json();
      setPipelineStep(11);
      setPipelineStatus('complete');
      setPipelineResult(data);

      // Parse some raw history rows for charting
      const text = await file.text();
      const lines = text.trim().split('\n');
      const headers = lines[0].split(',').map((h) => h.trim());
      const parsedRows = lines.slice(1, 100).map((l, idx) => {
        const cols = l.split(',');
        const obj = { index: idx };
        headers.forEach((h, i) => (obj[h] = cols[i]?.trim()));
        return obj;
      });
      setRawHistoryRows(parsedRows);
    } catch (err) {
      clearInterval(stepInterval);
      setPipelineStatus('error');
      setErrorMsg(err.message || 'Error connecting to FastAPI backend.');
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar apiOnline={apiOnline} checkingApi={checkingApi} onRefreshHealth={checkHealth} />

      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '24px', width: '100%', flex: 1 }}>
        {/* Error Alert Box */}
        {errorMsg && (
          <div
            style={{
              padding: '16px',
              background: 'rgba(244, 63, 94, 0.12)',
              border: '1px solid rgba(244, 63, 94, 0.35)',
              borderRadius: 'var(--radius-md)',
              color: '#fb7185',
              marginBottom: '24px',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
            }}
          >
            <AlertCircle size={20} style={{ flexShrink: 0 }} />
            <div>
              <p style={{ fontWeight: 600, fontSize: '0.92rem' }}>Pipeline Execution Error</p>
              <p style={{ fontSize: '0.82rem', marginTop: '2px', color: '#fecdd3' }}>{errorMsg}</p>
            </div>
          </div>
        )}

        {/* 1. Upload & Ingestion Section */}
        <UploadSection
          file={file}
          setFile={setFile}
          horizon={horizon}
          setHorizon={setHorizon}
          onRunPipeline={handleRunPipeline}
          isLoading={pipelineStatus === 'running'}
          selectedPreset={selectedPreset}
          onSelectPreset={handleSelectPreset}
        />

        {/* 2. Pipeline Execution Stepper */}
        {(pipelineStatus === 'running' || pipelineStatus === 'complete') && (
          <PipelineStepper status={pipelineStatus} currentStep={pipelineStep} />
        )}

        {/* 3. Output Dashboards (When Complete) */}
        {pipelineResult && (
          <>
            {/* Dataset Intelligence Overview */}
            <DatasetIntelligenceCard
              profile={pipelineResult.dataset_profile}
              schema={pipelineResult.schema}
              frequency={pipelineResult.frequency}
              validation={pipelineResult.validation}
            />

            {/* Candidate Models Benchmarking & Selected Model Card */}
            <ModelLeaderboard
              modelResults={pipelineResult.model_results}
              selectedModel={pipelineResult.selected_model}
              testMetrics={pipelineResult.test_metrics}
              modelPath={pipelineResult.model_path}
            />

            {/* Interactive Forecast Chart */}
            <ForecastChart
              historyData={rawHistoryRows}
              forecastData={pipelineResult.forecast}
              selectedModel={pipelineResult.selected_model}
            />

            {/* Step-by-Step Forecast Table & CSV Export */}
            <ForecastTable
              forecastData={pipelineResult.forecast}
              selectedModel={pipelineResult.selected_model}
              datasetName={file?.name?.replace('.csv', '')}
            />
          </>
        )}
      </main>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--border-color)', padding: '20px 24px', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
        <p>VoltCast AI • Production Time-Series Machine Learning Pipeline (FastAPI + React)</p>
      </footer>
    </div>
  );
}
