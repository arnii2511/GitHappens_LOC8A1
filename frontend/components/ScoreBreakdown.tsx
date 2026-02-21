'use client';

import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import {
  ShieldCheck, Package, Globe, Activity, Scale, TrendingUp,
  Repeat, FileCheck, CreditCard, UserCheck, History,
} from 'lucide-react';
import type { ScoreFactors, TrustComponents, MatchReason } from '@/lib/types';

interface ScoreBreakdownProps {
  factors: ScoreFactors;
  trustComponents: TrustComponents;
  reasons: MatchReason[];
  matchScore: number;
  trustScore: number;
  finalScore: number;
}

// Factor label map
const factorMeta: Record<keyof ScoreFactors, { label: string; icon: typeof Package }> = {
  productFit: { label: 'Product Fit', icon: Package },
  geographyFit: { label: 'Geography', icon: Globe },
  buyerActivity: { label: 'Activity', icon: Activity },
  scaleFit: { label: 'Scale', icon: Scale },
  marketTrend: { label: 'Market Trend', icon: TrendingUp },
  tradeFrequency: { label: 'Trade Freq.', icon: Repeat },
};

// Trust component label map
const trustMeta: Record<keyof TrustComponents, { label: string; shortLabel: string; icon: typeof ShieldCheck }> = {
  registrationVerification: { label: 'Registration Verification', shortLabel: 'RV', icon: UserCheck },
  tradeHistory: { label: 'Trade History', shortLabel: 'TH', icon: History },
  documentationFidelity: { label: 'Documentation Fidelity', shortLabel: 'DF', icon: FileCheck },
  paymentBehavior: { label: 'Payment Behavior', shortLabel: 'PB', icon: CreditCard },
};

function getColor(value: number) {
  if (value >= 90) return '#10b981';
  if (value >= 80) return '#0ea5e9';
  if (value >= 70) return '#f59e0b';
  return '#ef4444';
}

function ScoreIndicator({ label, value, max = 100 }: { label: string; value: number; max?: number }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-xs text-slate-400 w-20 flex-shrink-0">{label}</span>
      <div className="flex-1 bg-slate-700/50 rounded-full h-1.5 overflow-hidden border border-slate-600/30">
        <div
          className="h-full rounded-full transition-all duration-500 ease-out"
          style={{ width: `${(value / max) * 100}%`, backgroundColor: getColor(value) }}
        />
      </div>
      <span className="text-xs font-bold text-white w-8 text-right">{value}</span>
    </div>
  );
}

export default function ScoreBreakdown({
  factors,
  trustComponents,
  reasons,
  matchScore,
  trustScore,
  finalScore,
}: ScoreBreakdownProps) {
  // Bar chart data
  const barData = (Object.keys(factorMeta) as (keyof ScoreFactors)[]).map((key) => ({
    name: factorMeta[key].label,
    value: factors[key],
  }));

  // Radar chart data
  const radarData = (Object.keys(factorMeta) as (keyof ScoreFactors)[]).map((key) => ({
    factor: factorMeta[key].label,
    score: factors[key],
    fullMark: 100,
  }));

  return (
    <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-5 sticky top-24 space-y-5">
      {/* Score Summary */}
      <div>
        <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-3">Score Summary</h3>
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600/30 text-center">
            <p className="text-lg font-bold text-white">{matchScore}</p>
            <p className="text-[10px] text-slate-400 uppercase tracking-wider">Match</p>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600/30 text-center">
            <p className="text-lg font-bold text-white">{trustScore}</p>
            <p className="text-[10px] text-slate-400 uppercase tracking-wider">Trust</p>
          </div>
          <div className="bg-sky-500/10 rounded-lg p-3 border border-sky-500/20 text-center">
            <p className="text-lg font-bold text-sky-400">{finalScore}</p>
            <p className="text-[10px] text-sky-400/70 uppercase tracking-wider">Final</p>
          </div>
        </div>
      </div>

      <Separator className="bg-slate-700/50" />

      {/* Radar Chart */}
      <div>
        <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-2">Factor Profile</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="#475569" strokeDasharray="3 3" />
              <PolarAngleAxis
                dataKey="factor"
                tick={{ fill: '#94a3b8', fontSize: 9 }}
              />
              <PolarRadiusAxis
                angle={30}
                domain={[0, 100]}
                tick={{ fill: '#64748b', fontSize: 8 }}
                axisLine={false}
              />
              <Radar
                name="Score"
                dataKey="score"
                stroke="#0ea5e9"
                fill="#0ea5e9"
                fillOpacity={0.2}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Factor Breakdown Bars */}
      <div>
        <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-3">Factor Breakdown</h3>
        <div className="h-44">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} layout="vertical" margin={{ left: 0, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} stroke="#64748b" fontSize={10} tickLine={false} />
              <YAxis
                type="category"
                dataKey="name"
                stroke="#94a3b8"
                fontSize={10}
                width={75}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  color: '#e2e8f0',
                  fontSize: '12px',
                }}
                formatter={(value: number) => [`${value}/100`, 'Score']}
              />
              <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={14}>
                {barData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getColor(entry.value)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <Separator className="bg-slate-700/50" />

      {/* Trust Components */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">Trust Profile</h3>
        </div>
        <div className="space-y-2.5">
          {(Object.keys(trustMeta) as (keyof TrustComponents)[]).map((key) => (
            <ScoreIndicator
              key={key}
              label={trustMeta[key].shortLabel}
              value={trustComponents[key]}
            />
          ))}
        </div>
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {(Object.keys(trustMeta) as (keyof TrustComponents)[]).map((key) => (
            <Badge
              key={key}
              variant="outline"
              className="bg-slate-700/30 border-slate-600/40 text-[10px] text-slate-400"
            >
              {trustMeta[key].shortLabel} = {trustMeta[key].label}
            </Badge>
          ))}
        </div>
      </div>

      <Separator className="bg-slate-700/50" />

      {/* Algorithm Info */}
      <div className="bg-slate-700/20 rounded-lg p-3 border border-slate-600/30">
        <p className="text-[11px] text-slate-400 leading-relaxed">
          <span className="font-semibold text-sky-400">Algorithm:</span>{' '}
          Multi-factor evaluation with real-time adaptation based on market signals, trade history, and trust verification. Scores are updated continuously via snapshot time-series.
        </p>
      </div>
    </Card>
  );
}
