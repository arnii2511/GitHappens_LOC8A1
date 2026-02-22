'use client';

import { useState, useCallback, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MessageSquare } from 'lucide-react';
import MatchingCards from '@/components/MatchingCards';
import Dashboard, { type DashboardStats, type DashboardTimelinePoint } from '@/components/Dashboard';
import Outreach from '@/components/Outreach';
import Connections from '@/components/Connections';
import Header from '@/components/Header';
import type { ConnectionRequest, AppNotification, SwipeAction } from '@/lib/types';

export default function Home() {
  const [activeTab, setActiveTab] = useState('matches');
  const [connections, setConnections] = useState<ConnectionRequest[]>([]);
  const [notifications, setNotifications] = useState<AppNotification[]>([]);
  const [accountType, setAccountType] = useState<'exporter' | 'buyer'>('exporter');
  const [dashboardStats, setDashboardStats] = useState<DashboardStats>({
    totalMatches: 0,
    connected: 0,
    skipped: 0,
    saved: 0,
    blocked: 0,
    avgScore: 0,
    timeline: [{ label: 'Start', matches: 0, connected: 0, skipped: 0 }],
    scoreDistribution: [
      { range: '90-100', count: 0, fill: '#10b981' },
      { range: '80-89', count: 0, fill: '#06b6d4' },
      { range: '70-79', count: 0, fill: '#f59e0b' },
      { range: '<70', count: 0, fill: '#ef4444' },
    ],
  });

  const unreadCount = notifications.filter((n) => !n.read).length;

  const loadProfile = useCallback(async () => {
    const res = await fetch('/api/profile', { cache: 'no-store' });
    if (!res.ok) return;
    const data = (await res.json()) as { profile?: { account_type?: string } };
    const t = String(data.profile?.account_type || '').toLowerCase();
    if (t === 'buyer' || t === 'exporter') setAccountType(t);
  }, []);

  const loadConnections = useCallback(async () => {
    const res = await fetch('/api/connections', { cache: 'no-store' });
    if (!res.ok) return;
    const data = (await res.json()) as { connections?: ConnectionRequest[] };
    setConnections(data.connections ?? []);
  }, []);

  const loadNotifications = useCallback(async () => {
    const res = await fetch('/api/notifications', { cache: 'no-store' });
    if (!res.ok) return;
    const data = (await res.json()) as { notifications?: AppNotification[] };
    setNotifications(data.notifications ?? []);
  }, []);

  const loadAnalytics = useCallback(async () => {
    const res = await fetch('/api/analytics/summary', { cache: 'no-store' });
    if (!res.ok) return;
    const data = (await res.json()) as DashboardStats;
    setDashboardStats({
      totalMatches: Number(data.totalMatches ?? 0),
      connected: Number(data.connected ?? 0),
      skipped: Number(data.skipped ?? 0),
      saved: Number(data.saved ?? 0),
      blocked: Number(data.blocked ?? 0),
      avgScore: Number(data.avgScore ?? 0),
      timeline: Array.isArray(data.timeline) && data.timeline.length > 0 ? data.timeline : [{ label: 'Start', matches: 0, connected: 0, skipped: 0 }],
      scoreDistribution:
        Array.isArray(data.scoreDistribution) && data.scoreDistribution.length > 0
          ? data.scoreDistribution
          : [
              { range: '90-100', count: 0, fill: '#10b981' },
              { range: '80-89', count: 0, fill: '#06b6d4' },
              { range: '70-79', count: 0, fill: '#f59e0b' },
              { range: '<70', count: 0, fill: '#ef4444' },
            ],
    });
  }, []);

  useEffect(() => {
    void Promise.all([loadProfile(), loadConnections(), loadNotifications(), loadAnalytics()]);
  }, [loadProfile, loadConnections, loadNotifications, loadAnalytics]);

  useEffect(() => {
    const id = setInterval(() => {
      void Promise.all([loadConnections(), loadNotifications(), loadAnalytics()]);
    }, 8000);
    return () => clearInterval(id);
  }, [loadConnections, loadNotifications, loadAnalytics]);

  // Called by MatchingCards when user swipes Connect
  const handleNewConnection = useCallback(async (connection: ConnectionRequest) => {
    const res = await fetch('/api/connections', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        matchId: connection.matchId,
        importerOrgId: connection.importerOrgId,
        importerOrgName: connection.importerOrgName,
        importerCountry: connection.importerCountry,
        importerIndustry: connection.importerIndustry,
        finalScore: connection.finalScore,
        note: connection.note,
      }),
    });
    if (!res.ok) return;
    await Promise.all([loadConnections(), loadNotifications()]);
  }, [loadConnections, loadNotifications]);

  // Notification bell click navigates to Connections tab
  const handleNotificationClick = useCallback(() => {
    setActiveTab('connections');
  }, []);

  const handleSwipeAction = useCallback(
    async (payload: { action: SwipeAction; finalScore: number }) => {
      await fetch('/api/analytics/events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: payload.action,
          finalScore: payload.finalScore,
        }),
      });
      await loadAnalytics();
    },
    [loadAnalytics]
  );

  return (
    <div className="min-h-screen bg-background">
      <Header
        notificationCount={unreadCount}
        onNotificationClick={handleNotificationClick}
      />
      
      <main className="container mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full max-w-lg grid-cols-4 mb-8 bg-slate-700/50 border border-slate-600">
            <TabsTrigger value="matches" className="data-[state=active]:bg-sky-600 data-[state=active]:text-white text-slate-300">
              Matches
            </TabsTrigger>
            <TabsTrigger value="connections" className="data-[state=active]:bg-sky-600 data-[state=active]:text-white text-slate-300 relative">
              <MessageSquare className="w-3.5 h-3.5 mr-1.5" />
              Connect
              {unreadCount > 0 && (
                <span className="absolute -top-1.5 -right-1.5 bg-sky-500 text-white text-[9px] font-bold rounded-full w-4 h-4 flex items-center justify-center">
                  {unreadCount > 9 ? '9+' : unreadCount}
                </span>
              )}
            </TabsTrigger>
            <TabsTrigger value="dashboard" className="data-[state=active]:bg-sky-600 data-[state=active]:text-white text-slate-300">
              Analytics
            </TabsTrigger>
            <TabsTrigger value="outreach" className="data-[state=active]:bg-sky-600 data-[state=active]:text-white text-slate-300">
              Outreach
            </TabsTrigger>
          </TabsList>

          <TabsContent value="matches" className="space-y-4">
            <MatchingCards
              onConnect={handleNewConnection}
              onSwipeAction={handleSwipeAction}
              accountType={accountType}
            />
          </TabsContent>

          <TabsContent value="connections" className="space-y-4">
            <Connections
              connections={connections}
              setConnections={setConnections}
              notifications={notifications}
              setNotifications={setNotifications}
              accountType={accountType}
            />
          </TabsContent>

          <TabsContent value="dashboard" className="space-y-4">
            <Dashboard stats={dashboardStats} />
          </TabsContent>

          <TabsContent value="outreach" className="space-y-4">
            <Outreach />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
