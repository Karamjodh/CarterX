import { useState, useMemo } from 'react'
import { ComposableMap, Geographies, Geography, ZoomableGroup } from 'react-simple-maps'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts'

const GEO_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json'
const COLORS  = ['#7F77DD', '#1D9E75', '#D85A30', '#BA7517', '#185FA5', '#D4537E']

// Revenue tier colours — solid, high contrast
const REVENUE_COLORS = [
  '#2D2693',  // tier 1 — darkest (top revenue)
  '#4338CA',  // tier 2
  '#6366F1',  // tier 3
  '#818CF8',  // tier 4
  '#A5B4FC',  // tier 5
  '#C7D2FE',  // tier 6 — lightest (lowest revenue with data)
]
const NO_DATA_COLOR  = '#E2E8F0'  // clear grey for countries with no data
const BORDER_COLOR   = '#FFFFFF'
const ACTIVE_COLOR   = '#1D9E75'  // green for selected country

const COUNTRY_NAME_MAP = {
  'United Kingdom': 'United Kingdom', 'Uk': 'United Kingdom',
  'Usa': 'United States of America', 'United States': 'United States of America', 'Us': 'United States of America',
  'Uae': 'United Arab Emirates', 'Eire': 'Ireland', 'Channel Islands': 'United Kingdom',
  'Rsa': 'South Africa', 'Korea': 'South Korea', 'Czech Republic': 'Czechia', 'Hong Kong': 'China',
}

function fmt(v) {
  if (v == null || isNaN(v)) return '—'
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`
  if (v >= 1e6) return `$${(v / 1e6).toFixed(2)}M`
  if (v >= 1e3) return `$${(v / 1e3).toFixed(1)}K`
  return `$${Math.round(v).toLocaleString()}`
}

function normalizeCountry(name) {
  if (!name) return name
  const titled = name.trim().split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ')
  return COUNTRY_NAME_MAP[titled] || titled
}

export default function GeoTab({ insights }) {
  const geo = insights?.geo_data

  const [selectedRegion, setSelectedRegion] = useState(null)
  const [hoverInfo, setHoverInfo] = useState(null) // { x, y, name, entry }
  const [zoom,   setZoom]   = useState(1)
  const [center, setCenter] = useState([0, 20])

  // All hooks before any early return
  const region_stats         = geo?.region_stats         || []
  const top_regions          = geo?.top_regions          || []
  const region_growth        = geo?.region_growth        || []
  const regional_products    = geo?.regional_products    || {}
  const market_concentration = geo?.market_concentration || {}
  const summary              = geo?.summary              || {}
  const geo_column           = geo?.geo_column           || ''

  const topRegions   = top_regions.slice(0, 10)
  const activeRegion = selectedRegion || topRegions[0]?.region || ''

  // Build lookup: normalised name → rank (0 = top revenue)
  const regionLookup = useMemo(() => {
    const map = {}
    region_stats.forEach((r, i) => { map[normalizeCountry(r.region)] = { ...r, rank: i } })
    return map
  }, [region_stats])

  const totalRegions = region_stats.length

  const growthData = useMemo(() => {
    const found = region_growth.find(r => r.region === activeRegion)
    return found?.monthly || []
  }, [region_growth, activeRegion])

  const selectedProducts = regional_products[activeRegion] || []

  // Early return after all hooks
  if (!geo || !geo.has_geo_data) {
    return (
      <div style={S.empty}>
        <div style={S.emptyIcon}>⊕</div>
        <h3 style={S.emptyTitle}>No geographic data found</h3>
        <p style={S.emptySub}>{geo?.summary?.message || 'Your dataset does not contain a geographic column.'}</p>
        <div style={S.emptyHint}>Add a column named: <code>country</code>, <code>region</code>, <code>state</code>, <code>city</code>, or <code>location</code></div>
      </div>
    )
  }

  const hhiColor = market_concentration.hhi > 2500 ? '#993C1D' : market_concentration.hhi > 1500 ? '#BA7517' : '#0F6E56'

  // Assign solid colour by revenue rank — 6 tiers
  function getCountryColor(geoName) {
    const entry = regionLookup[geoName]
    if (!entry) return NO_DATA_COLOR
    const tier = Math.min(5, Math.floor((entry.rank / Math.max(totalRegions, 1)) * 6))
    return REVENUE_COLORS[tier]
  }

  return (
    <div>

      {/* ── KPI cards ── */}
      <div style={S.cards}>
        {[
          { label: 'Regions analyzed',    value: summary.total_regions,                       color: '#534AB7', bg: '#EEEDFE' },
          { label: 'Top region',          value: summary.top_region,                          color: '#0F6E56', bg: '#E1F5EE' },
          { label: 'Top region revenue',  value: fmt(summary.top_region_revenue),             color: '#0F6E56', bg: '#E1F5EE' },
          { label: 'Top region share',    value: `${summary.top_region_share}%`,              color: '#534AB7', bg: '#EEEDFE' },
          { label: 'Concentration (HHI)', value: market_concentration.hhi?.toLocaleString(), color: hhiColor,  bg: '#F7F6F3' },
          { label: 'Market type',         value: market_concentration.label,                  color: hhiColor,  bg: '#F7F6F3' },
        ].map(({ label, value, color, bg }) => (
          <div key={label} style={{ ...S.card, background: bg }}>
            <div style={S.cardLabel}>{label}</div>
            <div style={{ ...S.cardVal, color }}>{value ?? '—'}</div>
          </div>
        ))}
      </div>

      {/* ── World map ── */}
      <div style={S.section}>
        <div style={S.mapHeader}>
          <div>
            <div style={S.secTitle}>Customer geography</div>
            <div style={S.secSub}>
              Darker = higher revenue · click a country to drill down · column: <code>{geo_column}</code>
            </div>
          </div>
          <div style={S.mapControls}>
            <button style={S.zoomBtn} onClick={() => setZoom(z => Math.min(z + 0.5, 8))}>+</button>
            <button style={S.zoomBtn} onClick={() => setZoom(z => Math.max(z - 0.5, 1))}>−</button>
            <button style={{ ...S.zoomBtn, fontSize: 11, width: 48 }} onClick={() => { setZoom(1); setCenter([0, 20]) }}>Reset</button>
          </div>
        </div>

        {/* Colour legend */}
        <div style={S.legend}>
          <span style={S.legendLabel}>No data</span>
          <div style={{ ...S.legendChip, background: NO_DATA_COLOR }} />
          {REVENUE_COLORS.slice().reverse().map((c, i) => (
            <div key={i} style={{ ...S.legendChip, background: c }} />
          ))}
          <span style={S.legendLabel}>Highest revenue</span>
          <div style={{ ...S.legendChip, background: ACTIVE_COLOR, marginLeft: 12 }} />
          <span style={S.legendLabel}>Selected</span>
        </div>

        <div style={S.mapWrap}>
          <ComposableMap projectionConfig={{ scale: 147 }} style={{ width: '100%', height: '100%' }}>
            <ZoomableGroup zoom={zoom} center={center}
              onMoveEnd={({ zoom: z, coordinates }) => { setZoom(z); setCenter(coordinates) }}
            >
              <Geographies geography={GEO_URL}>
                {({ geographies }) =>
                  geographies.map(g => {
                    const geoName  = g.properties.name
                    const entry    = regionLookup[geoName]
                    const isActive = entry && normalizeCountry(activeRegion) === geoName
                    const fill     = isActive ? ACTIVE_COLOR : getCountryColor(geoName)
                    return (
                      <Geography
                        key={g.rsmKey}
                        geography={g}
                        fill={fill}
                        stroke={BORDER_COLOR}
                        strokeWidth={0.6}
                        style={{
                          default: { outline: 'none' },
                          hover:   { outline: 'none', fill: entry ? '#534AB7' : '#CBD5E1', cursor: entry ? 'pointer' : 'default', transition: 'fill 0.15s' },
                          pressed: { outline: 'none' },
                        }}
                        onMouseEnter={() => setHoverInfo({ x: 0, y: 0, geoName, entry })}
                        onMouseMove={(evt) => setHoverInfo({ x: evt.clientX, y: evt.clientY, geoName, entry })}
                        onMouseLeave={() => setHoverInfo(null)}
                        onClick={() => { if (entry) setSelectedRegion(entry.region) }}
                      />
                    )
                  })
                }
              </Geographies>
            </ZoomableGroup>
          </ComposableMap>
        </div>

        {hoverInfo && (
          <div style={{ ...S.tooltip, left: hoverInfo.x + 14, top: hoverInfo.y + 14 }}>
            <div style={S.tooltipName}>{hoverInfo.geoName}</div>
            {hoverInfo.entry ? (
              <>
                {hoverInfo.entry.total_revenue != null && (
                  <div>Revenue: {fmt(hoverInfo.entry.total_revenue)}</div>
                )}
                {hoverInfo.entry.revenue_share_pct != null && (
                  <div>Revenue contribution: {hoverInfo.entry.revenue_share_pct}%</div>
                )}
                {hoverInfo.entry.unique_customers != null && (
                  <div>Customers: {hoverInfo.entry.unique_customers.toLocaleString()}</div>
                )}
              </>
            ) : (
              <div style={{ color: '#94A3B8' }}>No data</div>
            )}
          </div>
        )}
      </div>

      {/* ── Market concentration ── (moved up) */}
      {market_concentration.hhi && (
        <div style={S.section}>
          <div style={S.secTitle}>Market concentration</div>
          <div style={S.secSub}>Herfindahl-Hirschman Index (HHI)</div>
          <div style={S.concRow}>
            <div style={S.concScore}>
              <div style={{ ...S.concNum, color: hhiColor }}>{market_concentration.hhi?.toLocaleString()}</div>
              <div style={{ ...S.concLabel, color: hhiColor }}>HHI Score</div>
            </div>
            <div style={S.concDesc}>
              <div style={S.concTitle}>{market_concentration.label}</div>
              <div style={S.concSub}>{market_concentration.description}</div>
              <div style={S.concBars}>
                {[{ label: '#1 region', val: market_concentration.top1_share }, { label: 'Top 3', val: market_concentration.top3_share }].map(({ label, val }, i) => (
                  <div key={label} style={S.concBarRow}>
                    <div style={S.concBarLabel}>{label}</div>
                    <div style={S.concBarTrack}><div style={{ ...S.concBarFill, width: `${val}%`, background: COLORS[i] }} /></div>
                    <div style={S.concBarVal}>{val}%</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Regional drill-down ── (moved up) */}
      {(growthData.length > 0 || selectedProducts.length > 0) && (
        <div style={S.section}>
          <div style={S.drillHeader}>
            <div style={S.secTitle}>Regional drill-down</div>
            <select value={activeRegion} onChange={e => setSelectedRegion(e.target.value)} style={S.select}>
              {topRegions.map(r => <option key={r.region} value={r.region}>{r.region}</option>)}
            </select>
          </div>
          <div style={S.drillGrid}>
            {growthData.length > 0 && (
              <div>
                <div style={S.drillLabel}>Monthly revenue — {activeRegion}</div>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={growthData} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" vertical={false} />
                    <XAxis dataKey="month" tick={{ fontSize: 10, fill: '#888780' }} tickFormatter={m => m?.slice(5)} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 10, fill: '#888780' }} axisLine={false} tickLine={false} tickFormatter={fmt} />
                    <Tooltip contentStyle={{ border: '0.5px solid #E8E6DF', borderRadius: 8, fontSize: 12 }} formatter={v => [fmt(v), 'Revenue']} />
                    <Line type="monotone" dataKey="total_revenue" stroke="#7F77DD" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            {selectedProducts.length > 0 && (
              <div>
                <div style={S.drillLabel}>Top products — {activeRegion}</div>
                <table style={S.table}>
                  <thead><tr><th style={S.th}>Product</th><th style={{ ...S.th, textAlign: 'right' }}>Revenue</th></tr></thead>
                  <tbody>
                    {selectedProducts.map((p, i) => (
                      <tr key={i} style={S.tr}>
                        <td style={S.td}>{p.product_name?.length > 30 ? p.product_name.slice(0, 30) + '…' : p.product_name}</td>
                        <td style={{ ...S.td, textAlign: 'right', fontWeight: 500 }}>{fmt(p.total_revenue)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Revenue bar chart ── */}
      {topRegions.length > 0 && (
        <div style={S.section}>
          <div style={S.secTitle}>Revenue by region</div>
          <div style={S.secSub}>Top {topRegions.length} regions</div>
          <ResponsiveContainer width="100%" height={Math.max(200, topRegions.length * 36 + 40)}>
            <BarChart data={topRegions} layout="vertical" barSize={18} margin={{ top: 8, right: 60, bottom: 0, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10, fill: '#888780' }} axisLine={false} tickLine={false} tickFormatter={fmt} />
              <YAxis type="category" dataKey="region" tick={{ fontSize: 11, fill: '#5F5E5A' }} width={120} axisLine={false} tickLine={false}
                tickFormatter={v => v?.length > 16 ? v.slice(0, 16) + '…' : v} />
              <Tooltip contentStyle={{ border: '0.5px solid #E8E6DF', borderRadius: 8, fontSize: 12 }}
                formatter={(v, _, props) => [fmt(v), `${props.payload.region} (${props.payload.revenue_share_pct}%)`]} />
              <Bar dataKey="total_revenue" fill="#7F77DD" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── All regions table ── */}
      {region_stats.length > 0 && (
        <div style={S.section}>
          <div style={S.secTitle}>All regions</div>
          <div style={S.secSub}>Click any row to drill down</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={S.th}>#</th>
                  <th style={S.th}>Region</th>
                  {region_stats[0]?.total_revenue    != null && <th style={{ ...S.th, textAlign: 'right' }}>Revenue</th>}
                  <th style={{ ...S.th, textAlign: 'right' }}>Share</th>
                  {region_stats[0]?.unique_customers != null && <th style={{ ...S.th, textAlign: 'right' }}>Customers</th>}
                  {region_stats[0]?.avg_order_value  != null && <th style={{ ...S.th, textAlign: 'right' }}>Avg Order</th>}
                  <th style={{ ...S.th, textAlign: 'right' }}>Cumulative</th>
                </tr>
              </thead>
              <tbody>
                {region_stats.map((r, i) => (
                  <tr key={r.region} onClick={() => setSelectedRegion(r.region)}
                    style={{ ...S.tr, background: r.region === activeRegion ? '#EEEDFE30' : i < 3 ? '#FAFAF8' : 'transparent', cursor: 'pointer' }}>
                    <td style={S.td}>
                      <span style={{ ...S.rank, background: i===0?'#EEEDFE':i===1?'#E1F5EE':i===2?'#FAECE7':'#F1EFE8', color: i===0?'#534AB7':i===1?'#0F6E56':i===2?'#993C1D':'#888780' }}>{i+1}</span>
                    </td>
                    <td style={{ ...S.td, fontWeight: 500, color: r.region === activeRegion ? '#534AB7' : '#1a1a1a' }}>{r.region}</td>
                    {r.total_revenue    != null && <td style={{ ...S.td, textAlign: 'right' }}>{fmt(r.total_revenue)}</td>}
                    <td style={{ ...S.td, textAlign: 'right' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 6 }}>
                        <div style={{ width: 40, height: 4, background: '#F1EFE8', borderRadius: 2 }}>
                          <div style={{ height: '100%', width: `${Math.min(100, r.revenue_share_pct||0)}%`, background: COLORS[i%COLORS.length], borderRadius: 2 }} />
                        </div>
                        {r.revenue_share_pct}%
                      </div>
                    </td>
                    {r.unique_customers != null && <td style={{ ...S.td, textAlign: 'right' }}>{r.unique_customers?.toLocaleString()}</td>}
                    {r.avg_order_value  != null && <td style={{ ...S.td, textAlign: 'right' }}>{fmt(r.avg_order_value)}</td>}
                    <td style={{ ...S.td, textAlign: 'right', color: '#888780' }}>{r.cumulative_share_pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

const S = {
  empty:        { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '4rem 2rem', textAlign: 'center' },
  emptyIcon:    { fontSize: 48, marginBottom: 16, color: '#D3D1C7' },
  emptyTitle:   { fontSize: 18, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', margin: 0, marginBottom: 8 },
  emptySub:     { fontSize: 14, color: '#5F5E5A', maxWidth: 400, lineHeight: 1.6, marginBottom: 12 },
  emptyHint:    { fontSize: 12, color: '#888780', background: '#F7F6F3', padding: '8px 16px', borderRadius: 8 },
  cards:        { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: '1rem' },
  card:         { borderRadius: 10, padding: '1rem' },
  cardLabel:    { fontSize: 11, color: '#5F5E5A', marginBottom: 4 },
  cardVal:      { fontSize: 16, fontWeight: 500 },
  section:      { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 12, padding: '1.25rem', marginBottom: '1rem' },
  mapHeader:    { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 },
  secTitle:     { fontSize: 15, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', marginBottom: 2 },
  secSub:       { fontSize: 12, color: '#888780', marginBottom: 8 },
  mapControls:  { display: 'flex', gap: 6 },
  zoomBtn:      { background: '#F7F6F3', border: '0.5px solid #E8E6DF', borderRadius: 6, width: 28, height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16, cursor: 'pointer', color: '#534AB7', fontWeight: 500 },
  legend:       { display: 'flex', alignItems: 'center', gap: 4, marginBottom: 10, flexWrap: 'wrap' },
  legendLabel:  { fontSize: 10, color: '#888780', whiteSpace: 'nowrap', marginRight: 2 },
  legendChip:   { width: 20, height: 12, borderRadius: 3 },
  mapWrap:      { width: '94%', margin: '0 auto', aspectRatio: '2 / 1', background: '#F8FAFC', borderRadius: 10, overflow: 'hidden', border: '2px solid #888780', touchAction: 'none' },
  tooltip:      { position: 'fixed', zIndex: 999, background: 'white', color: '#1a1a1a', border: '0.5px solid #E8E6DF', borderRadius: 8, fontSize: 12, padding: '8px 10px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)', pointerEvents: 'none', lineHeight: 1.6 },
  tooltipName:  { fontWeight: 600, marginBottom: 2 },
  concRow:      { display: 'flex', gap: 24, alignItems: 'flex-start', marginTop: 12 },
  concScore:    { background: '#EEEDFE', borderRadius: 10, padding: '1rem 1.25rem', textAlign: 'center', flexShrink: 0 },
  concNum:      { fontSize: 24, fontWeight: 500 },
  concLabel:    { fontSize: 11, marginTop: 2 },
  concDesc:     { flex: 1 },
  concTitle:    { fontSize: 14, fontWeight: 500, color: '#1a1a1a', marginBottom: 6 },
  concSub:      { fontSize: 12, color: '#888780', lineHeight: 1.6, marginBottom: 12 },
  concBars:     { display: 'flex', flexDirection: 'column', gap: 8 },
  concBarRow:   { display: 'flex', alignItems: 'center', gap: 10 },
  concBarLabel: { fontSize: 12, color: '#5F5E5A', width: 80, flexShrink: 0 },
  concBarTrack: { flex: 1, height: 8, background: '#F1EFE8', borderRadius: 4, overflow: 'hidden' },
  concBarFill:  { height: '100%', borderRadius: 4 },
  concBarVal:   { fontSize: 12, fontWeight: 500, width: 36, textAlign: 'right' },
  table:        { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th:           { textAlign: 'left', fontWeight: 500, fontSize: 11, color: '#888780', padding: '6px 0', borderBottom: '0.5px solid #E8E6DF' },
  tr:           { borderBottom: '0.5px solid #E8E6DF' },
  td:           { padding: '9px 0', color: '#5F5E5A', verticalAlign: 'middle' },
  rank:         { display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 22, height: 22, borderRadius: 6, fontSize: 11, fontWeight: 500 },
  drillHeader:  { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 },
  select:       { fontSize: 12, padding: '5px 10px', borderRadius: 8, border: '0.5px solid #D3D1C7', background: 'white' },
  drillGrid:    { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 },
  drillLabel:   { fontSize: 12, fontWeight: 500, color: '#5F5E5A', marginBottom: 10 },
}