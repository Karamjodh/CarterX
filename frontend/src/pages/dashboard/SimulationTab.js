import { useState, useEffect, useRef } from 'react'
import { Chart, registerables } from 'chart.js'
Chart.register(...registerables)

const PALETTE = ['#7F77DD', '#1D9E75', '#D85A30', '#BA7517', '#185FA5', '#D4537E']

function fmt(n) {
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`
  return `$${Math.round(n).toLocaleString()}`
}

export default function SimulationTab({ insights }) {
  const clusters   = insights.cluster_profiles || []
  const tsnePoints = insights.tsne_data || []
  const summary    = insights.summary || {}

  const [discount,  setDiscount]  = useState(15)
  const [horizon,   setHorizon]   = useState(90)
  const [targetSeg, setTargetSeg] = useState('all')

  const revenueChartRef = useRef(null)
  const revenueChart    = useRef(null)
  const segmentChartRef = useRef(null)
  const segmentChart    = useRef(null)
  const canvasRef       = useRef(null)

  const totalCusts  = clusters.reduce((s, c) => s + c.size, 0)
  const targetClust = targetSeg === 'all' ? clusters : clusters.filter(c => c.label === targetSeg)
  const targetCusts = targetClust.reduce((s, c) => s + c.size, 0)
  const targetPct   = totalCusts > 0 ? targetCusts / totalCusts : 1
  const totalRev    = summary.total_revenue || 0
  const dailyRate   = totalRev / 365
  const volUplift   = 1.5 * (discount / 100)
  const revMulti    = (1 - discount / 100) * (1 + volUplift)
  const baseProj    = Math.round(dailyRate * horizon * targetPct)
  const simProj     = Math.round(baseProj * revMulti)
  const liftPct     = baseProj > 0 ? +((simProj - baseProj) / baseProj * 100).toFixed(1) : 0
  const discCost    = Math.round(targetCusts * (summary.avg_order_value || 0) * (discount / 100) * 0.35)
  const roi         = discCost > 0 ? +((simProj - baseProj - discCost) / discCost * 100).toFixed(0) : 0

  function buildTimelineData() {
    const points = 8
    const labels = [], baseData = [], simData = []
    for (let i = 1; i <= points; i++) {
      const days = Math.round((horizon / points) * i)
      labels.push(`Day ${days}`)
      const base = Math.round(dailyRate * days * targetPct)
      baseData.push(base)
      simData.push(Math.round(base * revMulti))
    }
    return { labels, baseData, simData }
  }

  function buildSegmentData() {
    return clusters.map((c, i) => {
      const segPct   = totalCusts > 0 ? c.size / totalCusts : 0
      const segBase  = Math.round(dailyRate * horizon * segPct)
      const isTarget = targetSeg === 'all' || c.label === targetSeg
      return {
        label: c.label, base: segBase,
        sim: isTarget ? Math.round(segBase * revMulti) : segBase,
        color: PALETTE[i % PALETTE.length], targeted: isTarget,
      }
    })
  }

  // Revenue chart — destroy on every re-run, recreate fresh
  useEffect(() => {
    const canvas = revenueChartRef.current
    if (!canvas) return

    // Destroy any existing chart on this canvas
    if (revenueChart.current) {
      revenueChart.current.destroy()
      revenueChart.current = null
    }

    const { labels, baseData, simData } = buildTimelineData()
    revenueChart.current = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Baseline',
            data: baseData,
            borderColor: '#B4B2A9',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 3,
            pointBackgroundColor: '#B4B2A9',
            tension: 0.3,
          },
          {
            label: 'With campaign',
            data: simData,
            borderColor: '#7F77DD',
            backgroundColor: 'rgba(127,119,221,0.08)',
            fill: true,
            borderWidth: 2.5,
            pointRadius: 4,
            pointBackgroundColor: '#7F77DD',
            tension: 0.3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#fff',
            borderColor: '#E8E6DF',
            borderWidth: 1,
            titleColor: '#1a1a1a',
            bodyColor: '#5F5E5A',
            padding: 10,
            callbacks: { label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.raw)}` },
          },
        },
        scales: {
          x: { grid: { color: '#F1EFE8' }, ticks: { color: '#888780', font: { size: 11 } } },
          y: {
            grid: { color: '#F1EFE8' },
            ticks: { color: '#888780', font: { size: 11 }, callback: v => fmt(v) },
          },
        },
      },
    })

    return () => {
      revenueChart.current?.destroy()
      revenueChart.current = null
    }
  }, [discount, horizon, targetSeg, totalRev, totalCusts])

  // Segment chart — same pattern
  useEffect(() => {
    const canvas = segmentChartRef.current
    if (!canvas) return

    if (segmentChart.current) {
      segmentChart.current.destroy()
      segmentChart.current = null
    }

    const segData = buildSegmentData()
    segmentChart.current = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: segData.map(d => d.label),
        datasets: [
          {
            label: 'Baseline',
            data: segData.map(d => d.base),
            backgroundColor: '#F1EFE8',
            borderColor: '#D3D1C7',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'With campaign',
            data: segData.map(d => d.sim),
            backgroundColor: segData.map(d => d.targeted ? d.color : '#D3D1C7'),
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#fff',
            borderColor: '#E8E6DF',
            borderWidth: 1,
            titleColor: '#1a1a1a',
            bodyColor: '#5F5E5A',
            padding: 10,
            callbacks: { label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.raw)}` },
          },
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#888780', font: { size: 11 } } },
          y: {
            grid: { color: '#F1EFE8' },
            ticks: { color: '#888780', font: { size: 11 }, callback: v => fmt(v) },
          },
        },
      },
    })

    return () => {
      segmentChart.current?.destroy()
      segmentChart.current = null
    }
  }, [discount, horizon, targetSeg, totalRev, totalCusts])

  // t-SNE canvas
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
      sx: pad + ((x - minX) / (maxX - minX)) * (W - 2 * pad),
      sy: pad + ((y - minY) / (maxY - minY)) * (H - 2 * pad),
    })

    ctx.strokeStyle = '#F1EFE8'; ctx.lineWidth = 0.5
    for (let i = 0; i <= 5; i++) {
      const x = pad + i * (W - 2 * pad) / 5
      const y = pad + i * (H - 2 * pad) / 5
      ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke()
    }

    tsnePoints.forEach(pt => {
      const { sx, sy } = toScreen(pt.x, pt.y)
      const isTarget = targetSeg === 'all' || pt.label === targetSeg
      ctx.beginPath()
      ctx.arc(sx, sy, isTarget ? 4 : 2.5, 0, Math.PI * 2)
      ctx.fillStyle = isTarget ? PALETTE[pt.cluster_id % PALETTE.length] : '#D3D1C7'
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
      ctx.fillStyle = col; ctx.globalAlpha = 0.12; ctx.fill()
      ctx.globalAlpha = 1
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke()
      ctx.font = '500 11px Inter, system-ui, sans-serif'
      ctx.fillStyle = col; ctx.textAlign = 'center'
      ctx.fillText(label, sx, sy - 15)
    })
  }, [tsnePoints, targetSeg, clusters])

  const liftPos = liftPct >= 0
  const roiPos  = roi >= 0

  return (
    <div>
      <div style={S.controlsCard}>
        <div style={S.controlsRow}>
          <div style={S.ctrlGroup}>
            <label style={S.ctrlLabel}>Target segment</label>
            <select value={targetSeg} onChange={e => setTargetSeg(e.target.value)} style={S.select}>
              <option value="all">All segments ({totalCusts.toLocaleString()} customers)</option>
              {clusters.map(c => (
                <option key={c.cluster_id} value={c.label}>{c.label} ({c.size.toLocaleString()})</option>
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
            <input type="range" min="30" max="365" step="30"
              value={horizon} onChange={e => setHorizon(Number(e.target.value))} style={{ width: '100%' }} />
            <div style={S.hints}><span>30d</span><span>365d</span></div>
          </div>
        </div>
        <div style={S.note}>
          Price elasticity −1.5 · {discount}% discount → +{Math.round(volUplift * 100)}% volume uplift over {horizon} days
        </div>
      </div>

      <div style={S.kpiRow}>
        {[
          { label: 'Targeted customers', value: targetCusts.toLocaleString(),           color: '#534AB7', bg: '#EEEDFE' },
          { label: 'Baseline revenue',   value: fmt(baseProj),                          color: '#0F6E56', bg: '#E1F5EE' },
          { label: 'Projected revenue',  value: fmt(simProj),                           color: '#0F6E56', bg: '#E1F5EE' },
          { label: 'Revenue lift',       value: `${liftPos ? '+' : ''}${liftPct}%`,    color: liftPos ? '#534AB7' : '#993C1D', bg: liftPos ? '#EEEDFE' : '#FAECE7' },
          { label: 'Discount cost',      value: fmt(discCost),                          color: '#993C1D', bg: '#FAECE7' },
          { label: 'Est. ROI',           value: `${roiPos ? '+' : ''}${roi}%`,         color: roiPos  ? '#534AB7' : '#993C1D', bg: roiPos  ? '#EEEDFE' : '#FAECE7' },
        ].map(({ label, value, color, bg }) => (
          <div key={label} style={{ ...S.kpi, background: bg }}>
            <div style={S.kpiLabel}>{label}</div>
            <div style={{ ...S.kpiVal, color }}>{value}</div>
          </div>
        ))}
      </div>

      <div style={S.card}>
        <div style={S.cardHead}>
          <div>
            <div style={S.cardTitle}>Revenue projection over time</div>
            <div style={S.cardSub}>Cumulative baseline vs. campaign over {horizon} days</div>
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

      <div style={S.card}>
        <div style={S.cardHead}>
          <div>
            <div style={S.cardTitle}>Segment impact</div>
            <div style={S.cardSub}>Revenue per segment — targeted segments highlighted</div>
          </div>
          <div style={S.legend}>
            <span style={S.legItem}><span style={{ ...S.dot, background: '#F1EFE8', border: '1px solid #D3D1C7' }} />Baseline</span>
            <span style={S.legItem}><span style={{ ...S.dot, background: '#7F77DD' }} />Campaign</span>
          </div>
        </div>
        <div style={{ position: 'relative', height: Math.max(160, clusters.length * 80 + 60) }}>
          <canvas ref={segmentChartRef} />
        </div>
      </div>

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
  controlsCard: { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 12, padding: '1.25rem', marginBottom: 12 },
  controlsRow:  { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 24, marginBottom: 12 },
  ctrlGroup:    { display: 'flex', flexDirection: 'column', gap: 6 },
  ctrlLabel:    { fontSize: 12, color: '#888780' },
  select:       { fontSize: 13, padding: '6px 10px', borderRadius: 8, border: '0.5px solid #D3D1C7', background: 'white', color: '#1a1a1a' },
  hints:        { display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#B4B2A9', marginTop: 2 },
  note:         { fontSize: 11, color: '#888780', background: '#F7F6F3', padding: '7px 10px', borderRadius: 8 },
  kpiRow:       { display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 8, marginBottom: 12 },
  kpi:          { borderRadius: 10, padding: '10px 12px' },
  kpiLabel:     { fontSize: 11, color: '#5F5E5A', marginBottom: 4 },
  kpiVal:       { fontSize: 17, fontWeight: 500 },
  card:         { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 12, padding: '1.25rem', marginBottom: 12 },
  cardHead:     { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 },
  cardTitle:    { fontSize: 15, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', marginBottom: 2 },
  cardSub:      { fontSize: 12, color: '#888780' },
  legend:       { display: 'flex', gap: 12, alignItems: 'center' },
  legItem:      { display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: '#5F5E5A' },
  dot:          { width: 10, height: 10, borderRadius: 2, display: 'inline-block', flexShrink: 0 },
}