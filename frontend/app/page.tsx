'use client';

import { useState, useCallback } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MessageSquare } from 'lucide-react';
import MatchingCards from '@/components/MatchingCards';
import Dashboard from '@/components/Dashboard';
import Outreach from '@/components/Outreach';
import Connections from '@/components/Connections';
import Header from '@/components/Header';
import type { ConnectionRequest, AppNotification } from '@/lib/types';
import { mockConnections, mockNotifications } from '@/lib/mock-data';

export default function Home() {
  const [activeTab, setActiveTab] = useState('matches');
  const [connections, setConnections] = useState<ConnectionRequest[]>(mockConnections);
  const [notifications, setNotifications] = useState<AppNotification[]>(mockNotifications);

  const unreadCount = notifications.filter((n) => !n.read).length;

  // Called by MatchingCards when user swipes Connect
  const handleNewConnection = useCallback((connection: ConnectionRequest) => {
    setConnections((prev) => {
      // Prevent duplicate connections to the same importer
      if (prev.some((c) => c.importerOrgId === connection.importerOrgId)) return prev;
      return [connection, ...prev];
    });

    // Add notification for connection sent
    const notif: AppNotification = {
      id: `notif-${Date.now()}`,
      type: 'connection_sent',
      title: 'Connection Sent',
      description: `Your connection request to ${connection.importerOrgName} was sent.`,
      connectionId: connection.id,
      read: false,
      createdAt: new Date().toISOString(),
    };
    setNotifications((prev) => [notif, ...prev]);
  }, []);

  // Notification bell click navigates to Connections tab
  const handleNotificationClick = useCallback(() => {
    setActiveTab('connections');
  }, []);

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
            <MatchingCards onConnect={handleNewConnection} />
          </TabsContent>

          <TabsContent value="connections" className="space-y-4">
            <Connections
              connections={connections}
              setConnections={setConnections}
              notifications={notifications}
              setNotifications={setNotifications}
            />
          </TabsContent>

          <TabsContent value="dashboard" className="space-y-4">
            <Dashboard />
          </TabsContent>

          <TabsContent value="outreach" className="space-y-4">
            <Outreach />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
