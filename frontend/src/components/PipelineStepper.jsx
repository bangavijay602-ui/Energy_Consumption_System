import React from 'react';
import { CheckCircle2, Loader2, Circle } from 'lucide-react';

export default function PipelineStepper({ status, currentStep }) {
  const steps = [
    { id: 1, name: 'Dataset Profiling' },
    { id: 2, name: 'Schema Detection' },
    { id: 3, name: 'Normalization' },
    { id: 4, name: 'Data Validation' },
    { id: 5, name: 'Frequency Detection' },
    { id: 6, name: 'Feature Engineering' },
    { id: 7, name: 'Chronological Split' },
    { id: 8, name: 'Model Benchmarking' },
    { id: 9, name: 'Model Selection' },
    { id: 10, name: 'History Retraining' },
    { id: 11, name: 'Recursive Forecasting' },
  ];

  return (
    <div className="glass-card" style={{ padding: '18px 24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <h3 style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Autonomous Pipeline Execution Flow
        </h3>
        {status === 'running' && (
          <span className="badge badge-amber" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} />
            Processing Pipeline...
          </span>
        )}
        {status === 'complete' && (
          <span className="badge badge-emerald">
            <CheckCircle2 size={12} /> Pipeline Complete
          </span>
        )}
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(105px, 1fr))',
        gap: '8px',
        alignItems: 'center'
      }}>
        {steps.map((s) => {
          const isDone = status === 'complete' || currentStep > s.id;
          const isCurrent = status === 'running' && currentStep === s.id;

          return (
            <div
              key={s.id}
              style={{
                padding: '8px 10px',
                borderRadius: 'var(--radius-sm)',
                background: isDone
                  ? 'rgba(16, 185, 129, 0.12)'
                  : isCurrent
                  ? 'rgba(6, 182, 212, 0.18)'
                  : 'rgba(255, 255, 255, 0.02)',
                border: isDone
                  ? '1px solid rgba(16, 185, 129, 0.35)'
                  : isCurrent
                  ? '1px solid #06b6d4'
                  : '1px solid var(--border-color)',
                display: 'flex',
                flexDirection: 'column',
                gap: '4px',
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                {isDone ? (
                  <CheckCircle2 size={13} color="#34d399" />
                ) : isCurrent ? (
                  <Loader2 size={13} color="#38bdf8" style={{ animation: 'spin 1s linear infinite' }} />
                ) : (
                  <Circle size={13} color="#64748b" />
                )}
                <span style={{ fontSize: '0.72rem', fontWeight: 600, color: isDone ? '#34d399' : isCurrent ? '#38bdf8' : 'var(--text-dim)' }}>
                  {s.id}.
                </span>
              </div>
              <span style={{
                fontSize: '0.75rem',
                fontWeight: 500,
                color: isDone ? 'var(--text-main)' : isCurrent ? '#38bdf8' : 'var(--text-dim)',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}>
                {s.name}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
