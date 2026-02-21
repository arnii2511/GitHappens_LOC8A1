'use client';

import { Card } from '@/components/ui/card';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart
} from 'recharts';
import { TrendingUp, Users, Target, Zap } from 'lucide-react';

const matchSuccessData = [
  { month: 'Jan', matches: 45, connected: 32, deals: 8 },
  { month: 'Feb', matches: 52, connected: 38, deals: 12 },
  { month: 'Mar', matches: 48, connected: 35, deals: 10 },
  { month: 'Apr', matches: 65, connected: 48, deals: 16 },
  { month: 'May', matches: 72, connected: 54, deals: 19 },
  { month: 'Jun', matches: 85, connected: 62, deals: 24 },
];

const scoreDistribution = [
  { range: '90-100', count: 24, fill: '#10b981' },
  { range: '80-89', count: 42, fill: '#06b6d4' },
  { range: '70-79', count: 28, fill: '#f59e0b' },
  { range: '60-69', count: 12, fill: '#ef4444' },
];

const geographicData = [
  { region: 'Asia', value: 35 },
  { region: 'Americas', value: 28 },
  { region: 'Europe', value: 22 },
  { region: 'MENA', value: 15 },
];

const COLORS = ['#06b6d4', '#10b981', '#f59e0b', '#ef4444'];

const KPICard = ({ icon: Icon, title, value, change }: any) => (
  <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-6">
    <div className="flex items-start justify-between">
      <div>
        <p className="text-slate-400 text-sm uppercase tracking-wider">{title}</p>
        <p className="text-3xl font-bold text-white mt-2">{value}</p>
        <p className="text-xs text-green-400 mt-2">
          <TrendingUp className="w-3 h-3 inline mr-1" />
          {change}
        </p>
      </div>
      <div className="bg-gradient-to-br from-blue-600/20 to-cyan-500/20 p-3 rounded-lg">
        <Icon className="w-6 h-6 text-cyan-400" />
      </div>
    </div>
  </Card>
);

export default function Dashboard() {
  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <KPICard icon={Target} title="Total Matches" value="342" change="↑ 23% vs last month" />
        <KPICard icon={Users} title="Connected" value="186" change="↑ 18% vs last month" />
        <KPICard icon={Zap} title="Deals Closed" value="47" change="↑ 31% vs last month" />
        <KPICard icon={TrendingUp} title="Avg Score" value="84.2" change="↑ 2.4 pts" />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Match Success Timeline */}
        <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-6 lg:col-span-2">
          <h3 className="text-lg font-bold text-white mb-4">Match Success Timeline</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={matchSuccessData}>
              <defs>
                <linearGradient id="colorMatches" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="month" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
              />
              <Legend />
              <Area type="monotone" dataKey="matches" stroke="#06b6d4" fillOpacity={1} fill="url(#colorMatches)" />
              <Area type="monotone" dataKey="connected" stroke="#10b981" fillOpacity={0.1} />
              <Area type="monotone" dataKey="deals" stroke="#f59e0b" fillOpacity={0.1} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        {/* Geographic Distribution */}
        <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-6">
          <h3 className="text-lg font-bold text-white mb-4">Geographic Mix</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={geographicData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ region, value }) => `${region} ${value}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {COLORS.map((color, index) => (
                  <Cell key={`cell-${index}`} fill={color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
                formatter={(value) => `${value}%`}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Score Distribution */}
      <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-6">
        <h3 className="text-lg font-bold text-white mb-4">Match Score Distribution</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={scoreDistribution}>
            <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
            <XAxis dataKey="range" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip 
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #475569',
                borderRadius: '8px',
                color: '#e2e8f0'
              }}
            />
            <Bar dataKey="count" radius={[8, 8, 0, 0]}>
              {scoreDistribution.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-6">
          <h3 className="text-lg font-bold text-white mb-3">Key Insights</h3>
          <ul className="space-y-3 text-slate-300 text-sm">
            <li>✓ Match accuracy improved 12% with latest algorithm update</li>
            <li>✓ Asia region shows strongest growth at 35% of portfolio</li>
            <li>✓ Average deal closure time reduced to 18 days</li>
            <li>✓ 84% of deals initiated from 80+ scored matches</li>
          </ul>
        </Card>

        <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-6">
          <h3 className="text-lg font-bold text-white mb-3">Recommendations</h3>
          <ul className="space-y-3 text-slate-300 text-sm">
            <li>→ Focus on Americas region for growth opportunity</li>
            <li>→ Increase product category diversity (3 new SKUs)</li>
            <li>→ Target mid-market buyers for higher conversion</li>
            <li>→ Optimize timelines for Q3 peak season</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
