import { useState, useEffect, useRef, useMemo } from 'react'
import { Chart, registerables } from 'chart.js'
Chart.register(...registerables)

const PALETTE = ['#7F77DD', '#1D9E75', '#D85A30', '#BA7517', '#185FA5', '#D4537E']

function fmt(n) {
  if (n == null || isNaN(n)) return '—'
  if (Math.abs(n) >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
  if (Math.abs(n) >= 1e6) return `$${(n / 1e6).toFixed(2)}M`
  if (Math.abs(n) >= 1e3) return `$${(n / 1e3).toFixed(1)}K`
  return `$${Math.round(n).toLocaleString()}`
}

// ── Elasticity engine ─────────────────────────────────────────────────────────
// Instead of hardcoded -1.5, we compute elasticity per segment based on
// how their avg_monetary compares to the overall average.
// High-value customers (Champions) are less price-sensitive → lower elasticity
// Low-value customers (Hibernating) are more price-sensitive → higher elasticity
function computeElasticity(segment, allSegments) {
  if (!segment || !allSegments?.length) return -1.2

  const avgMonetary = allSegments.reduce((s, c) => s + c.avg_monetary, 0) / allSegments.length
  const segMonetary = segment.avg_monetary ?? avgMonetary

  // Normalize: high monetary → low elasticity, low monetary → high elasticity
  // Range: -0.6 (Champions) to -2.0 (Lost Customers)
  const ratio = avgMonetary > 0 ? segMonetary / avgMonetary : 1
  const elasticity = -0.6 - (1.4 * Math.max(0, Math.min(1, 1 - (ratio - 0.5))))
  return Math.round(elasticity * 100) / 100
}

// ── Build simulation projection ───────────────────────────────────────────────
function buildSimulation({ forecastData, clusters, targetSeg, discount, horizon, summary }) {
  const hasRealForecast = forecastData?.has_date_data && forecastData?.forecast?.length > 0

  // Get target segment object
  const targetCluster = targetSeg === 'all'
    ? null
    : clusters.find(c => c.label === targetSeg)

  const targetPct = targetSeg === 'all'
    ? 1
    : (targetCluster?.size ?? 0) / Math.max(1, clusters.reduce((s, c) => s + c.size, 0))

  // Compute elasticity for target segment
  const elasticity = computeElasticity(targetCluster, clusters)
  const volUplift  = Math.abs(elasticity) * (discount / 100)
  const priceEffect = 1 - discount / 100
  const revMulti   = priceEffect * (1 + volUplift)

  // ── Build baseline points ──────────────────────────────────────────────────
  let baselinePoints = []

  if (hasRealForecast) {
    // Use real LSTM forecast — slice to horizon
    baselinePoints = forecastData.forecast
      .slice(0, horizon)
      .map(f => ({ date: f.date, value: f.predicted }))
  } else {
    // Fallback: flat daily rate from summary revenue
    const totalRev  = summary?.total_revenue ?? 0
    const dailyRate = totalRev / 365
    const today     = new Date()
    for (let i = 0; i < horizon; i++) {
      const d = new Date(today)
      d.setDate(today.getDate() + i)
      baselinePoints.push({
        date:  d.toISOString().split('T')[0],
        value: dailyRate,
      })
    }
  }

  // ── Sample to ~40 points for chart performance ─────────────────────────────
  const step = Math.max(1, Math.floor(baselinePoints.length / 40))
  const sampled = baselinePoints.filter((_, i) => i % step === 0 || i === baselinePoints.length - 1)

  // ── Build chart data ───────────────────────────────────────────────────────
  const chartData = sampled.map(pt => {
    const base    = pt.value
    const baseAdj = base * targetPct           // only targeted customers
    const sim     = baseAdj * revMulti         // with campaign
    const nonTarget = base * (1 - targetPct)   // untargeted stays flat
    return {
      date:     pt.date,
      baseline: Math.round(base),
      scenario: Math.round(sim + nonTarget),
    }
  })

  // ── KPIs ──────────────────────────────────────────────────────────────────
  const baseTotal = baselinePoints.reduce((s, p) => s + p.value, 0)
  const simTotal  = baselinePoints.reduce((s, p) => {
    const adj = p.value * targetPct
    return s + (adj * revMulti) + (p.value * (1 - targetPct))
  }, 0)

  const targetCustomers = targetSeg === 'all'
    ? clusters.reduce((s, c) => s + c.size, 0)
    : (targetCluster?.size ?? 0)

  const avgOrderValue = summary?.avg_order_value ?? 0
  const discountCost  = Math.round(targetCustomers * avgOrderValue * (discount / 100) * 0.35)
  const lift          = baseTotal > 0 ? ((simTotal - baseTotal) / baseTotal * 100) : 0
  const roi           = discountCost > 0 ? ((simTotal - baseTotal - discountCost) / discountCost * 100) : 0

  // Break-even day — first day cumulative sim > cumulative base + cost
  let breakEvenDay = null
  let cumBase = 0, cumSim = 0
  for (let i = 0; i < baselinePoints.length; i++) {
    const pt   = baselinePoints[i]
    const adj  = pt.value * targetPct
    cumBase   += pt.value
    cumSim    += (adj * revMulti) + (pt.value * (1 - targetPct))
    if (cumSim - cumBase >= discountCost && breakEvenDay === null) {
      breakEvenDay = i + 1
    }
  }

  return {
    chartData,
    baseTotal:       Math.round(baseTotal),
    simTotal:        Math.round(simTotal),
    lift:            Math.round(lift * 10) / 10,
    discountCost,
    roi:             Math.round(roi),
    targetCustomers,
    elasticity,
    volUplift:       Math.round(volUplift * 100),
    breakEvenDay,
    hasRealForecast,
  }
}

export default function SimulationTab({ insights }) {
  const clusters   = insights.cluster_profiles || []
  const tsnePoints = insights.tsne_data        || []
  const summary    = insights.summary          || {}
  const forecastData = insights.forecast_data  || null

  const [discount,  setDiscount]  = useState(15)
  const [horizon,   setHorizon]   = useState(90)
  const [targetSeg, setTargetSeg] = useState('all')

  const revenueChartRef = useRef(null)
  const revenueChart    = useRef(null)
  const segmentChartRef = useRef(null)
  const segmentChart    = useRef(null)
  const canvasRef       = useRef(null)

  // ── Compute simulation ────────────────────────────────────────────────────
  const sim = useMemo(() => buildSimulation({
    forecastData, clusters, targetSeg, discount, horizon, summary
  }), [forecastData, clusters, targetSeg, discount, horizon, summary])

  const totalCusts = clusters.reduce((s, c) => s + c.size, 0)
  const liftPos    = sim.lift >= 0
  const roiPos     = sim.roi  >= 0

  // ── Revenue timeline chart ────────────────────────────────────────────────
  useEffect(() => {
    const canvas = revenueChartRef.current
    if (!canvas || !sim.chartData.length) return

    revenueChart.current?.destroy()
    revenueChart.current = null

    const labels   = sim.chartData.map(d => d.date.slice(5)) // MM-DD
    const baseData = sim.chartData.map(d => d.baseline)
    const simData  = sim.chartData.map(d => d.scenario)

    const allVals = [...baseData, ...simData]
    const minVal  = Math.min(...allVals)
    const maxVal  = Math.max(...allVals)
    const pad     = (maxVal - minVal) * 0.15 || maxVal * 0.05

    revenueChart.current = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label:           'Baseline',
            data:            baseData,
            borderColor:     '#B4B2A9',
            backgroundColor: 'transparent',
            borderWidth:     2,
            borderDash:      [5, 4],
            pointRadius:     0,
            tension:         0.3,
          },
          {
            label:           'With campaign',
            data:            simData,
            borderColor:     '#7F77DD',
            backgroundColor: 'rgba(127,119,221,0.10)',
            fill:            '-1',
            borderWidth:     2.5,
            pointRadius:     0,
            tension:         0.3,
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        interaction:         { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#fff',
            borderColor:     '#E8E6DF',
            borderWidth:     1,
            titleColor:      '#1a1a1a',
            bodyColor:       '#5F5E5A',
            padding:         10,
            callbacks: {
              label:     ctx  => ` ${ctx.dataset.label}: ${fmt(ctx.raw)}`,
              afterBody: items => {
                if (items.length < 2) return ''
                const diff = items[1].raw - items[0].raw
                return `\nLift: ${diff >= 0 ? '+' : ''}${fmt(diff)}`
              },
            },
          },
        },
        scales: {
          x: {
            grid:  { color: '#F1EFE8' },
            ticks: { color: '#888780', font: { size: 10 }, maxTicksLimit: 8 },
          },
          y: {
            min:   Math.max(0, minVal - pad),
            max:   maxVal + pad,
            grid:  { color: '#F1EFE8' },
            ticks: { color: '#888780', font: { size: 10 }, callback: v => fmt(v) },
          },
        },
      },
    })

    return () => { revenueChart.current?.destroy(); revenueChart.current = null }
  }, [sim])

  // ── Segment impact chart ──────────────────────────────────────────────────
  useEffect(() => {
    const canvas = segmentChartRef.current
    if (!canvas || !clusters.length) return

    segmentChart.current?.destroy()
    segmentChart.current = null

    const totalRev  = summary.total_revenue ?? 0
    const dailyRate = totalRev / 365

    const segData = clusters.map((c, i) => {
      const segPct   = totalCusts > 0 ? c.size / totalCusts : 0
      const segBase  = sim.hasRealForecast
        ? (forecastData.forecast.slice(0, horizon).reduce((s, f) => s + f.predicted, 0) * segPct)
        : Math.round(dailyRate * horizon * segPct)
      const isTarget = targetSeg === 'all' || c.label === targetSeg
      const elasticity = computeElasticity(c, clusters)
      const volUp    = Math.abs(elasticity) * (discount / 100)
      const multi    = (1 - discount / 100) * (1 + volUp)
      return {
        label:    c.label,
        base:     Math.round(segBase),
        sim:      isTarget ? Math.round(segBase * multi) : Math.round(segBase),
        color:    PALETTE[i % PALETTE.length],
        targeted: isTarget,
      }
    })

    segmentChart.current = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: segData.map(d => d.label),
        datasets: [
          {
            label:           'Baseline',
            data:            segData.map(d => d.base),
            backgroundColor: '#F1EFE8',
            borderColor:     '#D3D1C7',
            borderWidth:     1,
            borderRadius:    4,
          },
          {
            label:           'With campaign',
            data:            segData.map(d => d.sim),
            backgroundColor: segData.map(d => d.targeted ? d.color : '#D3D1C7'),
            borderRadius:    4,
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#fff',
            borderColor:     '#E8E6DF',
            borderWidth:     1,
            titleColor:      '#1a1a1a',
            bodyColor:       '#5F5E5A',
            padding:         10,
            callbacks: { label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.raw)}` },
          },
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#888780', font: { size: 10 } } },
          y: { grid: { color: '#F1EFE8' }, ticks: { color: '#888780', font: { size: 10 }, callback: v => fmt(v) } },
        },
      },
    })

    return () => { segmentChart.current?.destroy(); segmentChart.current = null }
  }, [sim, clusters, discount, horizon, targetSeg])

  // ── t-SNE canvas (unchanged) ──────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || tsnePoints.length === 0) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)

    const xs = tsnePoints.map(p => p.x), ys = tsnePoints.map(p => p.y)
    const minX = Math.min(...xs), maxX = Math.max(...xs)
    const minY = Math.min(...ys), maxY = Math.max(...ys)
    const pad = 32
    const toScreen = (x, y) => ({
      sx: pad + ((x - minX) / (maxX - minX || 1)) * (W - 2 * pad),
      sy: pad + ((y - minY) / (maxY - minY || 1)) * (H - 2 * pad),
    })

    ctx.strokeStyle = '#F1EFE8'; ctx.lineWidth = 0.5
    for (let i = 0; i <= 5; i++) {
      const x = pad + i * (W - 2 * pad) / 5
      const y = pad + i * (H - 2 * pad) / 5
      ctx.beginPath(); ctx.moveTo(x, pad);     ctx.lineTo(x, H - pad); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(pad, y);     ctx.lineTo(W - pad, y); ctx.stroke()
    }

    tsnePoints.forEach(pt => {
      const { sx, sy } = toScreen(pt.x, pt.y)
      const isTarget = targetSeg === 'all' || pt.label === targetSeg
      ctx.beginPath()
      ctx.arc(sx, sy, isTarget ? 4 : 2.5, 0, Math.PI * 2)
      ctx.fillStyle   = isTarget ? PALETTE[pt.cluster_id % PALETTE.length] : '#D3D1C7'
      ctx.globalAlpha = isTarget ? 0.85 : 0.25
      ctx.fill()
      ctx.globalAlpha = 1
    })

    const clusterGroups = {}
    tsnePoints.forEach(pt => {
      if (!clusterGroups[pt.label]) clusterGroups[pt.label] = []
      clusterGroups[pt.label].push(pt)
    })
    const labelMap = {}
    clusters.forEach((c, i) => { labelMap[c.label] = i })

    Object.entries(clusterGroups).forEach(([label, pts]) => {
      const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length
      const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length
      const { sx, sy } = toScreen(cx, cy)
      const col = PALETTE[(labelMap[label] ?? 0) % PALETTE.length]
      ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI * 2)
      ctx.fillStyle   = col; ctx.globalAlpha = 0.12; ctx.fill()
      ctx.globalAlpha = 1
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke()
      ctx.font        = '500 11px Inter, system-ui, sans-serif'
      ctx.fillStyle   = col; ctx.textAlign = 'center'
      ctx.fillText(label, sx, sy - 15)
    })
  }, [tsnePoints, targetSeg, clusters])

  return (
    <div>

      {/* ── Forecast source notice ────────────────────────────────────── */}
      {sim.hasRealForecast ? (
        <div style={S.noticeLSTM}>
          <span>🧠</span>
          <div>
            <strong>LSTM-powered baseline</strong> — simulation uses real forecast data.
            Elasticity is computed per segment from RFM monetary values.
          </div>
        </div>
      ) : (
        <div style={S.noticeFallback}>
          <span>⚡</span>
          <div>
            No date data in this dataset — using average daily revenue as baseline.
            Upload transactional data with dates for LSTM-powered simulation.
          </div>
        </div>
      )}

      {/* ── Controls ─────────────────────────────────────────────────── */}
      <div style={S.controlsCard}>
        <div style={S.controlsRow}>
          <div style={S.ctrlGroup}>
            <label style={S.ctrlLabel}>Target segment</label>
            <select value={targetSeg} onChange={e => setTargetSeg(e.target.value)} style={S.select}>
              <option value="all">All segments ({totalCusts.toLocaleString()} customers)</option>
              {clusters.map(c => (
                <option key={c.cluster_id} value={c.label}>
                  {c.label} ({c.size.toLocaleString()})
                </option>
              ))}
            </select>
          </div>
          <div style={S.ctrlGroup}>
            <label style={S.ctrlLabel}>Discount — <strong>{discount}%</strong></label>
            <input type="range" min="0" max="50" step="1"
              value={discount} onChange={e => setDiscount(Number(e.target.value))} style={{ width: '100%' }} />
            <div style={S.hints}><span>0%</span><span>50%</span></div>
          </div>
          <div style={S.ctrlGroup}>
            <label style={S.ctrlLabel}>Time horizon — <strong>{horizon} days</strong></label>
            <input type="range" min="30" max="180" step="30"
              value={horizon} onChange={e => setHorizon(Number(e.target.value))} style={{ width: '100%' }} />
            <div style={S.hints}><span>30d</span><span>180d</span></div>
          </div>
        </div>
        <div style={S.note}>
          Elasticity for {targetSeg === 'all' ? 'all segments' : targetSeg}: <strong>{sim.elasticity}</strong>
          &nbsp;·&nbsp; {discount}% discount → +{sim.volUplift}% volume uplift over {horizon} days
          {sim.breakEvenDay && ` · Break-even: day ${sim.breakEvenDay}`}
        </div>
      </div>

      {/* ── KPI cards ────────────────────────────────────────────────── */}
      <div style={S.kpiRow}>
        {[
          { label: 'Targeted customers', value: sim.targetCustomers.toLocaleString(),             color: '#534AB7', bg: '#EEEDFE' },
          { label: 'Baseline revenue',   value: fmt(sim.baseTotal),                               color: '#0F6E56', bg: '#E1F5EE' },
          { label: 'Projected revenue',  value: fmt(sim.simTotal),                                color: '#0F6E56', bg: '#E1F5EE' },
          { label: 'Revenue lift',       value: `${liftPos ? '+' : ''}${sim.lift}%`,             color: liftPos ? '#534AB7' : '#993C1D', bg: liftPos ? '#EEEDFE' : '#FAECE7' },
          { label: 'Discount cost',      value: fmt(sim.discountCost),                            color: '#993C1D', bg: '#FAECE7' },
          { label: 'Est. ROI',           value: `${roiPos ? '+' : ''}${sim.roi}%`,               color: roiPos  ? '#534AB7' : '#993C1D', bg: roiPos  ? '#EEEDFE' : '#FAECE7' },
        ].map(({ label, value, color, bg }) => (
          <div key={label} style={{ ...S.kpi, background: bg }}>
            <div style={S.kpiLabel}>{label}</div>
            <div style={{ ...S.kpiVal, color }}>{value}</div>
          </div>
        ))}
      </div>

      {/* ── Revenue projection chart ──────────────────────────────────── */}
      <div style={S.card}>
        <div style={S.cardHead}>
          <div>
            <div style={S.cardTitle}>Revenue projection over time</div>
            <div style={S.cardSub}>
              {sim.hasRealForecast
                ? 'LSTM forecast baseline vs campaign scenario'
                : 'Average daily rate baseline vs campaign scenario'}
            </div>
          </div>
          <div style={S.legend}>
            <span style={S.legItem}><span style={{ ...S.dot, background: '#B4B2A9' }} />Baseline</span>
            <span style={S.legItem}><span style={{ ...S.dot, background: '#7F77DD' }} />With campaign</span>
          </div>
        </div>
        <div style={{ position: 'relative', height: 260 }}>
          <canvas ref={revenueChartRef} />
        </div>
      </div>

      {/* ── Segment impact chart ──────────────────────────────────────── */}
      <div style={S.card}>
        <div style={S.cardHead}>
          <div>
            <div style={S.cardTitle}>Segment impact</div>
            <div style={S.cardSub}>Revenue per segment — targeted segments highlighted · elasticity varies by segment value</div>
          </div>
          <div style={S.legend}>
            <span style={S.legItem}><span style={{ ...S.dot, background: '#F1EFE8', border: '1px solid #D3D1C7' }} />Baseline</span>
            <span style={S.legItem}><span style={{ ...S.dot, background: '#7F77DD' }} />Campaign</span>
          </div>
        </div>
        <div style={{ position: 'relative', height: Math.max(160, clusters.length * 60 + 60) }}>
          <canvas ref={segmentChartRef} />
        </div>
      </div>

      {/* ── t-SNE cluster map (unchanged) ────────────────────────────── */}
      {tsnePoints.length > 0 && (
        <div style={S.card}>
          <div style={S.cardHead}>
            <div>
              <div style={S.cardTitle}>Customer cluster map</div>
              <div style={S.cardSub}>t-SNE projection — targeted segment highlighted</div>
            </div>
            <div style={S.legend}>
              {clusters.map((c, i) => (
                <span key={c.cluster_id} style={S.legItem}>
                  <span style={{ ...S.dot, background: PALETTE[i % PALETTE.length] }} />{c.label}
                </span>
              ))}
            </div>
          </div>
          <canvas ref={canvasRef} width={860} height={340}
            style={{ width: '100%', height: 'auto', borderRadius: 8, border: '0.5px solid #F1EFE8', display: 'block' }} />
        </div>
      )}
    </div>
  )
}

const S = {
  noticeLSTM:    { background: '#E1F5EE', border: '0.5px solid #1D9E75', borderRadius: 10, padding: '10px 14px', marginBottom: 12, display: 'flex', gap: 10, alignItems: 'flex-start', fontSize: 12, color: '#0F6E56' },
  noticeFallback:{ background: '#FAEEDA', border: '0.5px solid #BA7517', borderRadius: 10, padding: '10px 14px', marginBottom: 12, display: 'flex', gap: 10, alignItems: 'flex-start', fontSize: 12, color: '#854F0B' },
  controlsCard:  { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 12, padding: '1.25rem', marginBottom: 12 },
  controlsRow:   { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 24, marginBottom: 12 },
  ctrlGroup:     { display: 'flex', flexDirection: 'column', gap: 6 },
  ctrlLabel:     { fontSize: 12, color: '#888780' },
  select:        { fontSize: 13, padding: '6px 10px', borderRadius: 8, border: '0.5px solid #D3D1C7', background: 'white', color: '#1a1a1a' },
  hints:         { display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#B4B2A9', marginTop: 2 },
  note:          { fontSize: 11, color: '#888780', background: '#F7F6F3', padding: '7px 10px', borderRadius: 8 },
  kpiRow:        { display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 8, marginBottom: 12 },
  kpi:           { borderRadius: 10, padding: '10px 12px' },
  kpiLabel:      { fontSize: 11, color: '#5F5E5A', marginBottom: 4 },
  kpiVal:        { fontSize: 17, fontWeight: 500 },
  card:          { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 12, padding: '1.25rem', marginBottom: 12 },
  cardHead:      { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 },
  cardTitle:     { fontSize: 15, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', marginBottom: 2 },
  cardSub:       { fontSize: 12, color: '#888780' },
  legend:        { display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' },
  legItem:       { display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: '#5F5E5A' },
  dot:           { width: 10, height: 10, borderRadius: 2, display: 'inline-block', flexShrink: 0 },
}