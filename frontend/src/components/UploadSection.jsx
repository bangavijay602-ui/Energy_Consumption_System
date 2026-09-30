import React, { useRef } from 'react';
import { UploadCloud, FileText, Sliders, Play, Sparkles, Check, AlertCircle } from 'lucide-react';

export default function UploadSection({
  file,
  setFile,
  horizon,
  setHorizon,
  onRunPipeline,
  isLoading,
  selectedPreset,
  onSelectPreset,
}) {
  const fileInputRef = useRef(null);

  const presets = [
    { id: 'dataset1', name: 'Dataset 1 (Hourly)', desc: 'DateTime, Energy_Usage', file: 'dataset1.csv' },
    { id: 'dataset2', name: 'Dataset 2 (30-min)', desc: 'timestamp, load (Grid Peak)', file: 'dataset2.csv' },
    { id: 'dataset3', name: 'Dataset 3 (15-min)', desc: 'reading_time, power, temp, hum', file: 'dataset3.csv' },
    { id: 'dataset4', name: 'Dataset 4 (Daily)', desc: 'date, electricity_demand', file: 'dataset4.csv' },
    { id: 'dataset5', name: 'Dataset 5 (Hourly + Temp)', desc: 'time, value, temp', file: 'dataset5.csv' },
  ];

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0];
      if (dropped.name.endsWith('.csv')) {
        setFile(dropped);
      } else {
        alert('Please upload a valid .csv file.');
      }
    }
  };

  return (
    <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <UploadCloud size={20} color="#10b981" />
            1. Ingest Time-Series Dataset
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Upload any energy CSV format — the pipeline automatically detects schema, cadence, and features.
          </p>
        </div>

        {/* Quick presets */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)', fontWeight: 600 }}>PRESETS:</span>
          {presets.map((p) => (
            <button
              key={p.id}
              onClick={() => onSelectPreset(p)}
              disabled={isLoading}
              style={{
                padding: '6px 12px',
                fontSize: '0.78rem',
                fontWeight: 500,
                borderRadius: 'var(--radius-sm)',
                border: selectedPreset === p.id ? '1px solid #10b981' : '1px solid var(--border-color)',
                background: selectedPreset === p.id ? 'rgba(16, 185, 129, 0.18)' : 'rgba(255, 255, 255, 0.04)',
                color: selectedPreset === p.id ? '#34d399' : 'var(--text-main)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
              title={p.desc}
            >
              {p.name}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
        {/* Dropzone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: '2px dashed rgba(255, 255, 255, 0.15)',
            borderRadius: 'var(--radius-md)',
            padding: '24px',
            textAlign: 'center',
            cursor: 'pointer',
            background: 'rgba(255, 255, 255, 0.02)',
            transition: 'all 0.2s ease',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            minHeight: '130px',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#10b981')}
          onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.15)')}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".csv"
            style={{ display: 'none' }}
          />
          <FileText size={32} color={file ? '#10b981' : '#64748b'} />
          {file ? (
            <div>
              <p style={{ fontWeight: 600, color: '#34d399', fontSize: '0.95rem' }}>{file.name}</p>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                {(file.size / 1024).toFixed(1)} KB • Ready for ingestion
              </p>
            </div>
          ) : (
            <div>
              <p style={{ fontWeight: 500, fontSize: '0.9rem' }}>
                Drag & drop user CSV here, or <span style={{ color: '#38bdf8', textDecoration: 'underline' }}>browse</span>
              </p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                Supports DateTime/timestamp, energy/load/demand, and optional weather
              </p>
            </div>
          )}
        </div>

        {/* Configuration & Action controls */}
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <label style={{ fontSize: '0.88rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Sliders size={16} color="#06b6d4" />
                Forecast Horizon (Multi-Step Steps):
              </label>
              <span className="badge badge-cyan" style={{ fontSize: '0.8rem' }}>
                {horizon} Steps Ahead
              </span>
            </div>
            <input
              type="range"
              min="6"
              max="168"
              step="6"
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              style={{
                width: '100%',
                accentColor: '#10b981',
                cursor: 'pointer',
                height: '6px',
              }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '4px' }}>
              <span>6 steps (Short)</span>
              <span>24 steps (1 Day for Hourly)</span>
              <span>168 steps (1 Week)</span>
            </div>
          </div>

          <button
            className="btn-primary"
            onClick={onRunPipeline}
            disabled={!file || isLoading}
            style={{ width: '100%', height: '46px' }}
          >
            {isLoading ? (
              <>
                <Sparkles size={18} style={{ animation: 'pulse 1.5s infinite' }} />
                Training & Forecasting Candidates...
              </>
            ) : (
              <>
                <Play size={18} fill="#042f2e" />
                Run Full Training & Forecasting Pipeline
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
