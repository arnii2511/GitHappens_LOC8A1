'use client';

import { Card } from '@/components/ui/card';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { TrendingUp, Users, Target, Zap } from 'lucide-react';
import type { ComponentType } from 'react';

export type DashboardTimelinePoint = {
  label: string;
  matches: number;
  connected: number;
  skipped: number;
};

export type DashboardScoreBucket = {
  range: string;
  count: number;
  fill: string;
};

export type DashboardStats = {
  totalMatches: number;
  connected: number;
  skipped: number;
  saved: number;
  blocked: number;
  avgScore: number;
  timeline: DashboardTimelinePoint[];
  scoreDistribution: DashboardScoreBucket[];
};

const geographicData = [
  { region: 'Asia', value: 35 },
  { region: 'Americas', value: 28 },
  { region: 'Europe', value: 22 },
  { region: 'MENA', value: 15 },
];

const COLORS = ['#06b6d4', '#10b981', '#f59e0b', '#ef4444'];

const KPICard = ({ icon: Icon, title, value, change }: { icon: ComponentType<{ className?: string }>; title: string; value: string; change: string }) => (
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

export default function Dashboard({ stats }: { stats: DashboardStats }) {
  const totalActions = stats.connected + stats.skipped + stats.saved + stats.blocked;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <KPICard icon={Target} title="Total Matches" value={String(stats.totalMatches)} change="Live from your swipes" />
        <KPICard icon={Users} title="Connected" value={String(stats.connected)} change="Right swipes" />
        <KPICard icon={Zap} title="Skipped" value={String(stats.skipped)} change="Left swipes" />
        <KPICard icon={TrendingUp} title="Avg Score" value={stats.avgScore.toFixed(1)} change={`${totalActions} actions tracked`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-6 lg:col-span-2">
          <h3 className="text-lg font-bold text-white mb-4">Session Swipe Timeline</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={stats.timeline}>
              <defs>
                <linearGradient id="colorMatches" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="label" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  color: '#e2e8f0',
                }}
              />
              <Area type="monotone" dataKey="matches" stroke="#06b6d4" fillOpacity={1} fill="url(#colorMatches)" />
              <Area type="monotone" dataKey="connected" stroke="#10b981" fillOpacity={0.1} />
              <Area type="monotone" dataKey="skipped" stroke="#f59e0b" fillOpacity={0.1} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

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
                  color: '#e2e8f0',
                }}
                formatter={(value) => `${value}%`}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-6">
        <h3 className="text-lg font-bold text-white mb-4">Score Distribution (Reviewed Cards)</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={stats.scoreDistribution}>
            <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
            <XAxis dataKey="range" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #475569',
                borderRadius: '8px',
                color: '#e2e8f0',
              }}
            />
            <Bar dataKey="count" radius={[8, 8, 0, 0]}>
              {stats.scoreDistribution.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
