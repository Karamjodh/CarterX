import { useState, useMemo } from 'react'
import {
  ResponsiveContainer, ComposedChart, Area, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts'

const HORIZON_OPTIONS = [
  { label: '30d',  days: 30  },
  { label: '60d',  days: 60  },
  { label: '90d',  days: 90  },
  { label: '180d', days: 180 },
]

function fmt(v) {
  if (v == null) return '—'
  if (v >= 1e6) return `$${(v / 1e6).toFixed(2)}M`
  if (v >= 1e3) return `$${(v / 1e3).toFixed(1)}K`
  return `$${Math.round(v).toLocaleString()}`
}

function fmtDate(dateStr) {
  const d = new Date(dateStr)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

// Custom tooltip
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={S.tooltip}>
      <div style={S.tooltipDate}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontSize: 12, marginTop: 2 }}>
          {p.name}: {fmt(p.value)}
        </div>
      ))}
    </div>
  )
}

export default function ForecastTab({ insights }) {
  const fd = insights.forecast_data

  const [horizon, setHorizon] = useState(90)

  // No forecast data at all
  if (!fd) {
    return (
      <div style={S.emptyState}>
        <div style={S.emptyIcon}>📈</div>
        <h3 style={S.emptyTitle}>Forecast not available</h3>
        <p style={S.emptySub}>Re-run the pipeline to generate a forecast.</p>
      </div>
    )
  }

  // No date data in dataset
  if (!fd.has_date_data || fd.model_used === 'none') {
    return (
      <div style={S.emptyState}>
        <div style={S.emptyIcon}>📅</div>
        <h3 style={S.emptyTitle}>No date data found</h3>
        <p style={S.emptySub}>
          {fd.warning || 'Upload transactional data with dates to enable forecasting.'}
        </p>
      </div>
    )
  }

  const history  = fd.history  || []
  const forecast = fd.forecast || []

  // Slice forecast to selected horizon
  const forecastSlice = useMemo(
    () => forecast.slice(0, horizon),
    [forecast, horizon]
  )

  // Build combined chart data:
  // History points get actual revenue, forecast points get predicted/lower/upper
  const historyChartData = history.map(h => ({
    date:   h.date,
    actual: h.revenue,
  }))

  // Sample forecast to ~30 points max so chart isn't sluggish
  const step = Math.max(1, Math.floor(forecastSlice.length / 30))
  const forecastChartData = forecastSlice
    .filter((_, i) => i % step === 0 || i === forecastSlice.length - 1)
    .map(f => ({
      date:      fmtDate(f.date),
      predicted: f.predicted,
      lower:     f.lower,
      upper:     f.upper,
    }))

  // KPIs
  const endForecast   = forecastSlice[forecastSlice.length - 1]
  const startForecast = forecastSlice[0]
  const lastActual    = history[history.length - 1]?.revenue ?? 0
  const forecastEnd   = endForecast?.predicted ?? 0
  const forecastStart = startForecast?.predicted ?? 0
  const totalForecast = forecastSlice.reduce((s, f) => s + f.predicted, 0)
  const growthPct     = forecastStart > 0
    ? (((forecastEnd - forecastStart) / forecastStart) * 100).toFixed(1)
    : '0.0'

  return (
    <div>

      {/* ── Header ──────────────────────────────────────────────────── */}
      <div style={S.header}>
        <div>
          <div style={S.modelBadge}>
            {fd.model_used === 'LSTM' ? '🧠 LSTM Model' : fd.model_used === 'Prophet' ? '📊 Prophet Model' : '📈 Linear Trend'}
          </div>
          {fd.mae > 0 && (
            <div style={S.maeLine}>
              Validation MAE: {fmt(fd.mae)} · Based on {history.length} months of data
            </div>
          )}
        </div>

        {/* Horizon selector */}
        <div style={S.horizonGroup}>
          <span style={S.horizonLabel}>Forecast horizon</span>
          {HORIZON_OPTIONS.map(opt => (
            <button
              key={opt.days}
              onClick={() => setHorizon(opt.days)}
              style={{
                ...S.horizonBtn,
                background:  horizon === opt.days ? '#EEEDFE' : 'white',
                color:       horizon === opt.days ? '#534AB7' : '#5F5E5A',
                borderColor: horizon === opt.days ? '#AFA9EC' : '#D3D1C7',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── KPI cards ───────────────────────────────────────────────── */}
      <div style={S.kpiRow}>
        {[
          { label: 'Last actual month',    value: fmt(lastActual),    color: '#534AB7', bg: '#EEEDFE' },
          { label: `Revenue in ${horizon}d`, value: fmt(totalForecast), color: '#0F6E56', bg: '#E1F5EE' },
          { label: 'Forecast end value',   value: fmt(forecastEnd),   color: '#0F6E56', bg: '#E1F5EE' },
          { label: 'Growth over period',   value: `${growthPct > 0 ? '+' : ''}${growthPct}%`,
            color: growthPct >= 0 ? '#534AB7' : '#993C1D',
            bg:    growthPct >= 0 ? '#EEEDFE' : '#FAECE7' },
        ].map(({ label, value, color, bg }) => (
          <div key={label} style={{ ...S.kpi, background: bg }}>
            <div style={S.kpiLabel}>{label}</div>
            <div style={{ ...S.kpiVal, color }}>{value}</div>
          </div>
        ))}
      </div>

      {/* ── Historical chart ─────────────────────────────────────────── */}
      <div style={S.card}>
        <div style={S.cardTitle}>Historical revenue</div>
        <div style={S.cardSub}>Monthly actuals used to train the model</div>
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={historyChartData} margin={{ top: 8, right: 16, bottom: 0, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: '#888780' }}
              tickFormatter={v => v.length > 7 ? v.slice(0, 7) : v}
              axisLine={false} tickLine={false}
              interval={Math.floor(history.length / 6)}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#888780' }}
              tickFormatter={fmt}
              axisLine={false} tickLine={false}
              width={60}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="actual"
              name="Revenue"
              stroke="#534AB7"
              strokeWidth={2}
              dot={{ r: 3, fill: '#534AB7' }}
              activeDot={{ r: 5 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Forecast chart ───────────────────────────────────────────── */}
      <div style={S.card}>
        <div style={S.cardTitle}>Revenue forecast — next {horizon} days</div>
        <div style={S.cardSub}>
          Predicted daily revenue with ±12% confidence band
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={forecastChartData} margin={{ top: 8, right: 16, bottom: 0, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: '#888780' }}
              axisLine={false} tickLine={false}
              interval={Math.floor(forecastChartData.length / 6)}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#888780' }}
              tickFormatter={fmt}
              axisLine={false} tickLine={false}
              width={60}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              formatter={v => <span style={{ fontSize: 12, color: '#5F5E5A' }}>{v}</span>}
            />

            {/* Confidence band */}
            <Area
              type="monotone"
              dataKey="upper"
              name="Upper bound"
              stroke="none"
              fill="#EEEDFE"
              fillOpacity={0.6}
              legendType="none"
            />
            <Area
              type="monotone"
              dataKey="lower"
              name="Lower bound"
              stroke="none"
              fill="#F7F6F3"
              fillOpacity={1}
              legendType="none"
            />

            {/* Forecast line */}
            <Line
              type="monotone"
              dataKey="predicted"
              name="Predicted revenue"
              stroke="#7F77DD"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Warning if any ──────────────────────────────────────────── */}
      {fd.warning && (
        <div style={S.warning}>
          ⚡ {fd.warning}
        </div>
      )}

    </div>
  )
}

const S = {
  // Header
  header:       { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12, marginBottom: 16 },
  modelBadge:   { display: 'inline-block', background: '#EEEDFE', color: '#534AB7', border: '0.5px solid #AFA9EC', borderRadius: 20, padding: '4px 12px', fontSize: 12, fontWeight: 500, marginBottom: 4 },
  maeLine:      { fontSize: 11, color: '#888780' },
  horizonGroup: { display: 'flex', alignItems: 'center', gap: 6 },
  horizonLabel: { fontSize: 12, color: '#888780' },
  horizonBtn:   { padding: '5px 12px', borderRadius: 8, border: '0.5px solid', fontSize: 12, cursor: 'pointer', transition: 'all 0.15s' },
  // KPIs
  kpiRow:       { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 12 },
  kpi:          { borderRadius: 10, padding: '10px 12px' },
  kpiLabel:     { fontSize: 11, color: '#5F5E5A', marginBottom: 4 },
  kpiVal:       { fontSize: 17, fontWeight: 500 },
  // Cards
  card:         { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 12, padding: '1.25rem', marginBottom: 12 },
  cardTitle:    { fontSize: 15, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', marginBottom: 2 },
  cardSub:      { fontSize: 12, color: '#888780', marginBottom: 16 },
  // Tooltip
  tooltip:      { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 8, padding: '10px 14px', fontSize: 13 },
  tooltipDate:  { fontWeight: 500, color: '#1a1a1a', marginBottom: 4 },
  // Warning
  warning:      { background: '#FAEEDA', border: '0.5px solid #BA7517', borderRadius: 10, padding: '10px 14px', fontSize: 12, color: '#854F0B' },
  // Empty state
  emptyState:   { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '80px 20px', gap: 12 },
  emptyIcon:    { fontSize: 40 },
  emptyTitle:   { fontSize: 18, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', margin: 0 },
  emptySub:     { fontSize: 13, color: '#888780', margin: 0, textAlign: 'center', maxWidth: 360 },
}