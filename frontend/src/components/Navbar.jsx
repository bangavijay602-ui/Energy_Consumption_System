import React from 'react';
import { Zap, Activity, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';

export default function Navbar({ apiOnline, checkingApi, onRefreshHealth }) {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-color)',
      background: 'rgba(10, 15, 29, 0.85)',
      backdropFilter: 'blur(12px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '16px 24px'
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        {/* Logo & Branding */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(16, 185, 129, 0.4)'
          }}>
            <Zap size={22} color="#042f2e" strokeWidth={2.5} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '8px' }}>
              VoltCast AI
              <span className="badge badge-emerald" style={{ fontSize: '0.65rem', padding: '2px 8px' }}>
                Production ML
              </span>
            </h1>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Autonomous Energy Consumption Forecasting Pipeline
            </p>
          </div>
        </div>

        {/* Backend Health Status Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div 
            onClick={onRefreshHealth}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 14px',
              borderRadius: 'var(--radius-full)',
              background: apiOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)',
              border: `1px solid ${apiOnline ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
              cursor: 'pointer',
              fontSize: '0.82rem',
              fontWeight: 500,
              color: apiOnline ? '#34d399' : '#fb7185',
              transition: 'all 0.2s ease'
            }}
            title="Click to re-check FastAPI connection"
          >
            {checkingApi ? (
              <RefreshCw size={14} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
            ) : apiOnline ? (
              <CheckCircle2 size={15} color="#10b981" />
            ) : (
              <XCircle size={15} color="#f43f5e" />
            )}
            <span>
              FastAPI: {apiOnline ? 'Connected (Port 8000)' : 'Offline / Standalone Mode'}
            </span>
          </div>
        </div>
      </div>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </header>
  );
}
