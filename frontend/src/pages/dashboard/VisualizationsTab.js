import {
  BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts'

const COLORS = ['#7F77DD','#1D9E75','#D85A30','#BA7517','#185FA5']

function fmtY(v) {
  if (v == null || isNaN(v)) return '$0'
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`
  if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`
  return `$${v}`
}

function safeNum(v) { return v == null || isNaN(v) ? 0 : v }

function EmptyCard({ title, sub, message, icon = '📊' }) {
  return (
    <div style={S.section}>
      <div style={S.secTitle}>{title}</div>
      <div style={S.secSub}>{sub}</div>
      <div style={S.emptyState}>
        <div style={S.emptyIcon}>{icon}</div>
        <div style={S.emptyMsg}>{message}</div>
      </div>
    </div>
  )
}

export default function VisualizationsTab({ insights }) {
  const trend      = insights?.trend_data || {}
  const monthly    = trend.monthly_revenue || []
  const topProds   = trend.top_products || []
  const catRevenue = trend.category_revenue || []
  const clusters   = insights?.cluster_profiles || []
  const segData    = clusters
    .map(c => ({ name: c.label || `Segment ${c.cluster_id}`, value: safeNum(c.size) }))
    .filter(d => d.value > 0)

  const hasMonthly    = monthly.length > 0
  const hasCatRevenue = catRevenue.length > 0
  const hasChartData  = hasMonthly || hasCatRevenue

  return (
    <div>
      <div style={S.twoCol}>

        {hasChartData ? (
          <div style={S.section}>
            <div style={S.secTitle}>{hasMonthly ? 'Monthly revenue' : 'Revenue by category'}</div>
            <div style={S.secSub}>{hasMonthly ? 'Full period trend' : 'Top categories by revenue'}</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={hasMonthly ? monthly : catRevenue} margin={{ top: 8, right: 8, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" vertical={false} />
                <XAxis
                  dataKey={hasMonthly ? 'month' : 'category'}
                  tick={{ fontSize: 9, fill: '#888780' }}
                  tickFormatter={v => v && v.length > 12 ? v.slice(0, 12) + '…' : v}
                  axisLine={false} tickLine={false}
                  interval={hasMonthly ? 'preserveStartEnd' : 0}
                />
                <YAxis tick={{ fontSize: 10, fill: '#888780' }} axisLine={false} tickLine={false} tickFormatter={fmtY} width={55} />
                <Tooltip
                  contentStyle={{ border: '0.5px solid #E8E6DF', borderRadius: 8, fontSize: 12 }}
                  formatter={v => [`$${safeNum(v).toLocaleString()}`, 'Revenue']}
                />
                <Bar dataKey={hasMonthly ? 'total_revenue' : 'revenue'} fill="#7F77DD" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <EmptyCard
            title="Revenue trends" sub="Monthly or category breakdown"
            message="No date or revenue data available. Upload transactional data with dates to see revenue trends."
          />
        )}

        {segData.length > 0 ? (
          <div style={S.section}>
            <div style={S.secTitle}>Customer segments</div>
            <div style={S.secSub}>Distribution by size</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={segData} cx="50%" cy="50%" outerRadius={80} innerRadius={40}
                  dataKey="value"
                  label={({ name, percent }) => percent > 0.05 ? `${name} ${(percent * 100).toFixed(0)}%` : ''}
                  labelLine={false} fontSize={11}
                >
                  {segData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip
                  contentStyle={{ border: '0.5px solid #E8E6DF', borderRadius: 8, fontSize: 12 }}
                  formatter={(v, name) => [v.toLocaleString() + ' customers', name]}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <EmptyCard title="Customer segments" sub="Distribution by size" message="No segment data available." />
        )}

      </div>

      {topProds.length > 0 ? (
        <div style={S.section}>
          <div style={S.secTitle}>Top products by revenue</div>
          <div style={S.secSub}>Ranked by total sales value</div>
          <ResponsiveContainer width="100%" height={Math.max(200, topProds.length * 36 + 40)}>
            <BarChart data={topProds} layout="vertical" barSize={18} margin={{ top: 8, right: 40, bottom: 0, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10, fill: '#888780' }} axisLine={false} tickLine={false} tickFormatter={fmtY} />
              <YAxis type="category" dataKey="product_name" tick={{ fontSize: 9, fill: '#5F5E5A' }} width={180} axisLine={false} tickLine={false} tickFormatter={v => v && v.length > 24 ? v.slice(0, 24) + '…' : v} />
              <Tooltip
                contentStyle={{ border: '0.5px solid #E8E6DF', borderRadius: 8, fontSize: 12 }}
                formatter={(v, _n, props) => [`$${safeNum(v).toLocaleString()}`, props.payload.product_name]}
              />
              <Bar dataKey="total_revenue" fill="#1D9E75" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <EmptyCard
          title="Top products by revenue" sub="Ranked by total sales value" icon="📦"
          message="No product data available. Make sure your file includes a product name or product ID column."
        />
      )}
    </div>
  )
}

const S = {
  twoCol:    { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 },
  section:   { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 12, padding: '1.25rem', marginBottom: 12 },
  secTitle:  { fontSize: 15, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', marginBottom: 2 },
  secSub:    { fontSize: 12, color: '#888780', marginBottom: 12 },
  emptyState:{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px 20px', gap: 10 },
  emptyIcon: { fontSize: 28, opacity: 0.4 },
  emptyMsg:  { fontSize: 12, color: '#B4B2A9', textAlign: 'center', maxWidth: 280, lineHeight: 1.5 },
}