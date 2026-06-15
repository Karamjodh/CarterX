import { useState, useEffect, useRef } from 'react'

const PALETTE = ['#7F77DD','#1D9E75','#D85A30','#BA7517','#185FA5','#D4537E']
const LIGHT   = ['#EEEDFE','#E1F5EE','#FAECE7','#FAEEDA','#E6F1FB','#FBEAF0']

export default function SimulationTab({ insights }) {
  const clusters    = insights.cluster_profiles || []
  const tsnePoints  = insights.tsne_data || []
  const summary     = insights.summary || {}

  const [discount,    setDiscount]    = useState(15)
  const [horizon,     setHorizon]     = useState(90)
  const [targetSeg,   setTargetSeg]   = useState('all')
  const [hoveredPt,   setHoveredPt]   = useState(null)
  const canvasRef = useRef(null)

  const totalCusts  = clusters.reduce((s,c) => s+c.size, 0)
  const targetClust = targetSeg === 'all' ? clusters : clusters.filter(c => c.label === targetSeg)
  const targetCusts = targetClust.reduce((s,c) => s+c.size, 0)
  const targetPct   = totalCusts > 0 ? targetCusts / totalCusts : 1
  const totalRev    = summary.total_revenue || 0
  const dailyRate   = totalRev / 365
  const volUplift   = 1.5 * (discount / 100)
  const revMulti    = (1 - discount / 100) * (1 + volUplift)
  const baseProj    = Math.round(dailyRate * horizon * targetPct)
  const simProj     = Math.round(baseProj * revMulti)
  const liftPct     = baseProj > 0 ? ((simProj - baseProj) / baseProj * 100).toFixed(1) : 0
  const discCost    = Math.round(targetCusts * (summary.avg_order_value || 0) * (discount/100) * 0.35)
  const roi         = discCost > 0 ? (((simProj - baseProj) - discCost) / discCost * 100).toFixed(0) : 0

  // Draw 2D cluster scatter on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || tsnePoints.length === 0) return
    const ctx    = canvas.getContext('2d')
    const W      = canvas.width
    const H      = canvas.height

    ctx.clearRect(0, 0, W, H)

    const xs   = tsnePoints.map(p => p.x)
    const ys   = tsnePoints.map(p => p.y)
    const minX = Math.min(...xs), maxX = Math.max(...xs)
    const minY = Math.min(...ys), maxY = Math.max(...ys)
    const pad  = 32

    const toScreen = (x, y) => ({
      sx: pad + ((x - minX) / (maxX - minX)) * (W - 2*pad),
      sy: pad + ((y - minY) / (maxY - minY)) * (H - 2*pad),
    })

    // Draw background grid
    ctx.strokeStyle = '#F1EFE8'
    ctx.lineWidth   = 0.5
    for (let i = 0; i <= 5; i++) {
      const x = pad + i * (W - 2*pad) / 5
      const y = pad + i * (H - 2*pad) / 5
      ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H-pad); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W-pad, y); ctx.stroke()
    }

    // Draw points
    tsnePoints.forEach(pt => {
      const { sx, sy } = toScreen(pt.x, pt.y)
      const isTarget = targetSeg === 'all' || pt.label === targetSeg
      const color    = PALETTE[pt.cluster_id % PALETTE.length]

      ctx.beginPath()
      ctx.arc(sx, sy, isTarget ? 4 : 2.5, 0, Math.PI * 2)
      ctx.fillStyle = isTarget ? color : '#D3D1C7'
      ctx.globalAlpha = isTarget ? 0.8 : 0.3
      ctx.fill()
      ctx.globalAlpha = 1
    })

    // Draw cluster centroids
    const labelMap = {}
    clusters.forEach((c,i) => { labelMap[c.label] = i })

    const clusterGroups = {}
    tsnePoints.forEach(pt => {
      if (!clusterGroups[pt.label]) clusterGroups[pt.label] = []
      clusterGroups[pt.label].push(pt)
    })

    Object.entries(clusterGroups).forEach(([label, pts]) => {
      const cx = pts.reduce((s,p) => s+p.x, 0) / pts.length
      const cy = pts.reduce((s,p) => s+p.y, 0) / pts.length
      const { sx, sy } = toScreen(cx, cy)
      const ci = labelMap[label] ?? 0

      // Centroid circle
      ctx.beginPath()
      ctx.arc(sx, sy, 8, 0, Math.PI * 2)
      ctx.fillStyle = PALETTE[ci % PALETTE.length]
      ctx.globalAlpha = 0.15
      ctx.fill()
      ctx.globalAlpha = 1
      ctx.strokeStyle = PALETTE[ci % PALETTE.length]
      ctx.lineWidth   = 1.5
      ctx.stroke()

      // Label
      ctx.font      = '500 11px Inter, sans-serif'
      ctx.fillStyle = PALETTE[ci % PALETTE.length]
      ctx.textAlign = 'center'
      ctx.fillText(label, sx, sy - 14)
    })

  }, [tsnePoints, targetSeg, clusters])

  const histData = clusters.map((c, i) => ({
    label: c.label,
    color: PALETTE[i % PALETTE.length],
    light: LIGHT[i % LIGHT.length],
    monetary:  c.avg_monetary,
    frequency: c.avg_frequency,
    recency:   c.avg_recency_days,
  }))
  const maxM = Math.max(...histData.map(d => d.monetary))
  const maxF = Math.max(...histData.map(d => d.frequency))
  const maxR = Math.max(...histData.map(d => d.recency))

  return (
    <div>

      {/* 2D Cluster map */}
      <div style={S.section}>
        <div style={S.secHeader}>
          <div>
            <div style={S.secTitle}>Customer cluster map</div>
            <div style={S.secSub}>2D projection of customer similarity (t-SNE)</div>
          </div>
          <select
            value={targetSeg}
            onChange={e => setTargetSeg(e.target.value)}
            style={S.select}
          >
            <option value="all">All segments</option>
            {clusters.map(c => (
              <option key={c.cluster_id} value={c.label}>{c.label}</option>
            ))}
          </select>
        </div>

        {tsnePoints.length > 0 ? (
          <canvas
            ref={canvasRef}
            width={860}
            height={380}
            style={S.canvas}
          />
        ) : (
          <div style={S.placeholder}>
            t-SNE data not available. Re-upload your file to generate the cluster map.
          </div>
        )}

        <div style={S.legend}>
          {clusters.map((c,i) => (
            <div key={c.cluster_id} style={S.legendItem}>
              <div style={{...S.legendDot, background: PALETTE[i % PALETTE.length]}} />
              <span style={S.legendLabel}>{c.label}</span>
              <span style={S.legendCount}>{c.size}</span>
            </div>
          ))}
        </div>
      </div>

      {/* RFM Histograms */}
      <div style={S.threeCol}>
        {[
          { key:'monetary',  label:'Avg spend ($)',    max:maxM, fmt: v => `$${Math.round(v).toLocaleString()}` },
          { key:'frequency', label:'Buy frequency (×)', max:maxF, fmt: v => `${v.toFixed(1)}×` },
          { key:'recency',   label:'Recency (days)',   max:maxR, fmt: v => `${Math.round(v)}d` },
        ].map(metric => (
          <div key={metric.key} style={S.section}>
            <div style={S.secTitle}>{metric.label}</div>
            <div style={{display:'flex', flexDirection:'column', gap:10, marginTop:12}}>
              {histData.map(d => (
                <div key={d.label}>
                  <div style={S.barHeader}>
                    <span style={{fontSize:12, color:'#5F5E5A'}}>{d.label}</span>
                    <span style={{fontSize:12, fontWeight:500, color:'#1a1a1a'}}>{metric.fmt(d[metric.key])}</span>
                  </div>
                  <div style={S.barTrack}>
                    <div style={{
                      height:'100%',
                      width:`${(d[metric.key]/metric.max)*100}%`,
                      background: d.color,
                      borderRadius:4,
                      transition:'width 0.4s ease',
                    }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Simulation controls */}
      <div style={S.section}>
        <div style={S.secTitle}>Campaign simulator</div>
        <div style={S.secSub}>Adjust parameters to project campaign impact</div>

        <div style={S.simGrid}>
          <div style={S.controls}>
            <div style={S.controlRow}>
              <label style={S.ctrlLabel}>Target segment</label>
              <select value={targetSeg} onChange={e => setTargetSeg(e.target.value)} style={{flex:1}}>
                <option value="all">All segments ({totalCusts} customers)</option>
                {clusters.map(c => (
                  <option key={c.cluster_id} value={c.label}>
                    {c.label} ({c.size} customers)
                  </option>
                ))}
              </select>
            </div>

            <div style={S.controlRow}>
              <label style={S.ctrlLabel}>Discount</label>
              <div style={S.sliderWrap}>
                <input type="range" min="0" max="50" step="1"
                  value={discount} onChange={e => setDiscount(Number(e.target.value))}
                  style={{flex:1}} />
                <span style={S.sliderVal}>{discount}%</span>
              </div>
            </div>

            <div style={S.controlRow}>
              <label style={S.ctrlLabel}>Time horizon</label>
              <div style={S.sliderWrap}>
                <input type="range" min="30" max="365" step="30"
                  value={horizon} onChange={e => setHorizon(Number(e.target.value))}
                  style={{flex:1}} />
                <span style={S.sliderVal}>{horizon}d</span>
              </div>
            </div>

            <div style={S.elasticNote}>
              Model: price elasticity −1.5 · {discount}% discount → +{(1.5*discount/100*100).toFixed(0)}% volume uplift
            </div>
          </div>

          <div style={S.kpiGrid}>
            {[
              ['Targeted customers', targetCusts,                    '#534AB7', '#EEEDFE'],
              ['Baseline revenue',   `$${baseProj.toLocaleString()}`, '#0F6E56', '#E1F5EE'],
              ['Projected revenue',  `$${simProj.toLocaleString()}`,  '#0F6E56', '#E1F5EE'],
              ['Revenue lift',       `+${liftPct}%`,                  '#534AB7', '#EEEDFE'],
              ['Est. discount cost', `$${discCost.toLocaleString()}`,  '#993C1D', '#FAECE7'],
              ['Est. ROI',           `${roi}%`,                        '#534AB7', '#EEEDFE'],
            ].map(([label, val, color, bg]) => (
              <div key={label} style={{...S.kpiCard, background:bg}}>
                <div style={S.kpiLabel}>{label}</div>
                <div style={{...S.kpiVal, color}}>{val}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Comparison bars */}
        <div style={{marginTop:24}}>
          <div style={S.compTitle}>Revenue projection comparison</div>
          {[
            { label:'Baseline (no campaign)', val:baseProj, color:'#D3D1C7', max:simProj },
            { label:'Simulated (with campaign)', val:simProj, color:'#7F77DD', max:simProj },
          ].map(bar => (
            <div key={bar.label} style={{marginBottom:12}}>
              <div style={S.barHeader}>
                <span style={{fontSize:13, color:'#5F5E5A'}}>{bar.label}</span>
                <span style={{fontSize:13, fontWeight:500}}>${bar.val.toLocaleString()}</span>
              </div>
              <div style={{...S.barTrack, height:20}}>
                <div style={{
                  height:'100%',
                  width:`${(bar.val/bar.max)*100}%`,
                  background: bar.color,
                  borderRadius:4,
                  transition:'width 0.4s ease',
                }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

const S = {
  section:     { background:'white', border:'0.5px solid #E8E6DF', borderRadius:12, padding:'1.25rem', marginBottom:'1rem' },
  secHeader:   { display:'flex', alignItems:'flex-start', justifyContent:'space-between', marginBottom:16 },
  secTitle:    { fontSize:15, fontFamily:"'Instrument Serif', serif", color:'#1a1a1a', marginBottom:2 },
  secSub:      { fontSize:12, color:'#888780' },
  select:      { fontSize:12, padding:'5px 10px', borderRadius:8, border:'0.5px solid #D3D1C7', background:'white' },
  canvas:      { width:'100%', height:'auto', borderRadius:8, border:'0.5px solid #E8E6DF', display:'block' },
  placeholder: { height:200, display:'flex', alignItems:'center', justifyContent:'center', fontSize:13, color:'#888780', background:'#F7F6F3', borderRadius:8 },
  legend:      { display:'flex', gap:16, flexWrap:'wrap', marginTop:12 },
  legendItem:  { display:'flex', alignItems:'center', gap:6 },
  legendDot:   { width:8, height:8, borderRadius:'50%' },
  legendLabel: { fontSize:12, color:'#5F5E5A' },
  legendCount: { fontSize:12, color:'#888780' },
  threeCol:    { display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:12, marginBottom:'1rem' },
  barHeader:   { display:'flex', justifyContent:'space-between', marginBottom:4 },
  barTrack:    { height:8, background:'#F1EFE8', borderRadius:4, overflow:'hidden' },
  simGrid:     { display:'grid', gridTemplateColumns:'1fr 1fr', gap:24, marginTop:16 },
  controls:    { display:'flex', flexDirection:'column', gap:16 },
  controlRow:  { display:'flex', alignItems:'center', gap:12 },
  ctrlLabel:   { fontSize:13, color:'#5F5E5A', width:120, flexShrink:0 },
  sliderWrap:  { display:'flex', alignItems:'center', gap:10, flex:1 },
  sliderVal:   { fontSize:13, fontWeight:500, color:'#1a1a1a', width:36, textAlign:'right' },
  elasticNote: { fontSize:11, color:'#888780', background:'#F7F6F3', padding:'8px 10px', borderRadius:8 },
  kpiGrid:     { display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 },
  kpiCard:     { borderRadius:8, padding:'10px 12px' },
  kpiLabel:    { fontSize:11, color:'#5F5E5A', marginBottom:4 },
  kpiVal:      { fontSize:18, fontWeight:500 },
  compTitle:   { fontSize:13, fontWeight:500, color:'#5F5E5A', marginBottom:12 },
}