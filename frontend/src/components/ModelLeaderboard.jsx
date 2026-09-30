import React from 'react';
import { Trophy, Award, TrendingUp, BarChart3, CheckCircle2 } from 'lucide-react';

export default function ModelLeaderboard({ modelResults, selectedModel, testMetrics, modelPath }) {
  if (!modelResults) return null;

  const sortedModels = Object.entries(modelResults)
    .map(([name, m]) => ({ name, ...m }))
    .sort((a, b) => a.mae - b.mae);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginBottom: '24px' }}>
      {/* Candidate Leaderboard */}
      <div className="glass-card" style={{ padding: '20px', gridColumn: 'span 2' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={18} color="#10b981" />
            <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>Candidate Model Validation Leaderboard</h3>
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
            Ranked by Lowest Validation MAE
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-dim)', fontSize: '0.75rem', textTransform: 'uppercase' }}>
                <th style={{ padding: '10px 12px' }}>Rank</th>
                <th style={{ padding: '10px 12px' }}>Model</th>
                <th style={{ padding: '10px 12px' }}>Val MAE</th>
                <th style={{ padding: '10px 12px' }}>Val RMSE</th>
                <th style={{ padding: '10px 12px' }}>MAPE</th>
                <th style={{ padding: '10px 12px' }}>WAPE</th>
                <th style={{ padding: '10px 12px' }}>R² Score</th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.map((m, idx) => {
                const isWinner = m.name === selectedModel;
                return (
                  <tr
                    key={m.name}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                      background: isWinner ? 'rgba(16, 185, 129, 0.08)' : 'transparent',
                      fontWeight: isWinner ? 600 : 400,
                    }}
                  >
                    <td style={{ padding: '12px' }}>
                      {idx === 0 ? (
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: '#fbbf24', fontWeight: 700 }}>
                          <Trophy size={14} /> #1
                        </span>
                      ) : (
                        `#${idx + 1}`
                      )}
                    </td>
                    <td style={{ padding: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ color: isWinner ? '#34d399' : 'var(--text-main)' }}>{m.name}</span>
                      {isWinner && <span className="badge badge-emerald" style={{ fontSize: '0.65rem' }}>Selected</span>}
                    </td>
                    <td style={{ padding: '12px', color: isWinner ? '#34d399' : 'inherit' }}>{m.mae?.toFixed(4)}</td>
                    <td style={{ padding: '12px' }}>{m.rmse?.toFixed(4)}</td>
                    <td style={{ padding: '12px' }}>{m.mape?.toFixed(2)}%</td>
                    <td style={{ padding: '12px' }}>{m.wape?.toFixed(2)}%</td>
                    <td style={{ padding: '12px', color: m.r2 > 0.8 ? '#34d399' : m.r2 > 0 ? '#fbbf24' : '#f43f5e' }}>
                      {m.r2?.toFixed(4)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Winner Champion Card */}
      <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
            <Award size={20} color="#fbbf24" />
            <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>Winner Performance</h3>
          </div>

          <div style={{
            background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%)',
            border: '1px solid rgba(16, 185, 129, 0.35)',
            borderRadius: 'var(--radius-md)',
            padding: '16px',
            marginBottom: '16px',
          }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>
              Retrained Champion
            </span>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#34d399', margin: '4px 0 8px' }}>
              {selectedModel}
            </h2>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Retrained on all historical observations (Train + Validation)
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 600 }}>HELD-OUT TEST MAE</span>
              <p style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc' }}>
                {testMetrics?.mae?.toFixed(4) || '—'}
              </p>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 600 }}>TEST RMSE</span>
              <p style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc' }}>
                {testMetrics?.rmse?.toFixed(4) || '—'}
              </p>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 600 }}>TEST MAPE</span>
              <p style={{ fontSize: '1.15rem', fontWeight: 700, color: '#34d399' }}>
                {testMetrics?.mape?.toFixed(2) || '—'}%
              </p>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 600 }}>TEST R² SCORE</span>
              <p style={{ fontSize: '1.15rem', fontWeight: 700, color: '#38bdf8' }}>
                {testMetrics?.r2?.toFixed(4) || '—'}
              </p>
            </div>
          </div>
        </div>

        {modelPath && (
          <div style={{ marginTop: '16px', fontSize: '0.72rem', color: 'var(--text-dim)', wordBreak: 'break-all' }}>
            <span style={{ fontWeight: 600 }}>Serialized Artifact:</span> {modelPath}
          </div>
        )}
      </div>
    </div>
  );
}
