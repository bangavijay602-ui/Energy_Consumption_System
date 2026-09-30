import React from 'react';
import { Database, Clock, ShieldCheck, AlertTriangle, Cpu, Layers } from 'lucide-react';

export default function DatasetIntelligenceCard({ profile, schema, frequency, validation }) {
  if (!profile || !schema || !frequency) return null;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', marginBottom: '24px' }}>
      {/* 1. Dataset Profiling */}
      <div className="glass-card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
          <Database size={18} color="#06b6d4" />
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Dataset Telemetry</h3>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>Observations</span>
            <p style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f8fafc' }}>{profile.num_rows?.toLocaleString()}</p>
          </div>
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>Columns</span>
            <p style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f8fafc' }}>{profile.num_columns}</p>
          </div>
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>Missing Cells</span>
            <p style={{ fontSize: '1.1rem', fontWeight: 600, color: profile.total_missing_cells > 0 ? '#fbbf24' : '#34d399' }}>
              {profile.missing_cell_percentage}%
            </p>
          </div>
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>Memory Usage</span>
            <p style={{ fontSize: '1.1rem', fontWeight: 600, color: '#38bdf8' }}>{profile.memory_usage_mb || 0.1} MB</p>
          </div>
        </div>
      </div>

      {/* 2. Detected Schema Mapping */}
      <div className="glass-card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
          <Layers size={18} color="#10b981" />
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Discovered Schema</h3>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {Object.entries(schema).map(([canon, raw]) => (
            <div
              key={canon}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '6px 10px',
                background: 'rgba(255, 255, 255, 0.03)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.82rem',
              }}
            >
              <span style={{ fontWeight: 600, color: canon === 'timestamp' ? '#38bdf8' : canon === 'consumption' ? '#34d399' : '#fbbf24' }}>
                {canon}
              </span>
              <span className="font-mono" style={{ color: 'var(--text-main)', background: 'rgba(0,0,0,0.3)', padding: '2px 8px', borderRadius: '4px' }}>
                '{raw}'
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* 3. Frequency & Validation */}
      <div className="glass-card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
          <Clock size={18} color="#8b5cf6" />
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Cadence & Validation</h3>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Sampling Cadence:</span>
            <span className="badge badge-purple">{frequency.name || frequency}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Periods / Day:</span>
            <span style={{ fontWeight: 600, fontSize: '0.88rem' }}>{frequency.periods_per_day || 24}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Integrity Check:</span>
            <span className="badge badge-emerald">
              <ShieldCheck size={12} /> Valid
            </span>
          </div>

          {validation?.warnings && validation.warnings.length > 0 && (
            <div style={{
              marginTop: '6px',
              padding: '8px 10px',
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid rgba(245, 158, 11, 0.25)',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.74rem',
              color: '#fbbf24',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '6px'
            }}>
              <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: '2px' }} />
              <span>{validation.warnings[0]}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
