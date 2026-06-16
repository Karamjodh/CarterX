import {
  BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts'

const COLORS = ['#7F77DD','#1D9E75','#D85A30','#BA7517','#185FA5']

function fmtY(v) {
  if (v >= 1e9) return `$${(v / 1e9).toFixed(0)}B`
  if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`
  if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`
  return `$${v}`
}

export default function VisualizationsTab({ insights }) {
  const trend      = insights.trend_data || {}
  const monthly    = trend.monthly_revenue || []
  const topProds   = trend.top_products || []
  const catRevenue = trend.category_revenue || []
  const clusters   = insights.cluster_profiles || []
  const segData    = clusters.map(c => ({ name: c.label, value: c.size }))

  const hasMonthly = monthly.length > 0

  return (
    <div>
      <div style={S.twoCol}>

        {/* Monthly revenue OR category revenue fallback */}
        <div style={S.section}>
          <div style={S.secTitle}>{hasMonthly ? 'Monthly revenue' : 'Revenue by category'}</div>
          <div style={S.secSub}>{hasMonthly ? 'Full year trend' : 'Top categories by revenue'}</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={hasMonthly ? monthly : catRevenue}
              margin={{ top: 8, right: 8, bottom: 20, left: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" vertical={false} />
              <XAxis
                dataKey={hasMonthly ? 'month' : 'category'}
                tick={{ fontSize: 9, fill: '#888780' }}
                tickFormatter={v => v.length > 12 ? v.slice(0, 12) + '…' : v}
                axisLine={false} tickLine={false}
                interval={0}
              />
              <YAxis
                tick={{ fontSize: 10, fill: '#888780' }} axisLine={false} tickLine={false}
                tickFormatter={fmtY}
                width={55}
              />
              <Tooltip
                contentStyle={{ border: '0.5px solid #E8E6DF', borderRadius: 8, fontSize: 12 }}
                formatter={v => [`$${v.toLocaleString()}`, 'Revenue']}
              />
              <Bar
                dataKey={hasMonthly ? 'total_revenue' : 'revenue'}
                fill="#7F77DD" radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Customer segments */}
        <div style={S.section}>
          <div style={S.secTitle}>Customer segments</div>
          <div style={S.secSub}>By size</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={segData} cx="50%" cy="50%" outerRadius={80} innerRadius={40}
                dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={false} fontSize={11}
              >
                {segData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={{ border: '0.5px solid #E8E6DF', borderRadius: 8, fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top products */}
      {topProds.length > 0 && (
        <div style={S.section}>
          <div style={S.secTitle}>Top products by revenue</div>
          <div style={S.secSub}>Ranked by total sales value</div>
          <ResponsiveContainer width="100%" height={topProds.length * 36 + 40}>
            <BarChart data={topProds} layout="vertical" barSize={18}
              margin={{ top: 8, right: 40, bottom: 0, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10, fill: '#888780' }}
                axisLine={false} tickLine={false}
                tickFormatter={fmtY} />
              <YAxis type="category" dataKey="product_name"
                tick={{ fontSize: 9, fill: '#5F5E5A' }} width={180}
                axisLine={false} tickLine={false}
                tickFormatter={v => v.length > 24 ? v.slice(0, 24) + '…' : v} />
              <Tooltip
                contentStyle={{ border: '0.5px solid #E8E6DF', borderRadius: 8, fontSize: 12 }}
                formatter={(v, _name, props) => [`$${v.toLocaleString()}`, props.payload.product_name]}
              />
              <Bar dataKey="total_revenue" fill="#1D9E75" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

const S = {
  twoCol:   { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 },
  section:  { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 12, padding: '1.25rem', marginBottom: 12 },
  secTitle: { fontSize: 15, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', marginBottom: 2 },
  secSub:   { fontSize: 12, color: '#888780', marginBottom: 12 },
}