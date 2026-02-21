'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Building2,
  Globe,
  Send,
  Clock,
  CheckCircle2,
  XCircle,
  MessageSquare,
  UserPlus,
  ArrowLeft,
  Lock,
  Search,
} from 'lucide-react';
import type {
  ConnectionRequest,
  ConnectionStatus,
  ChatMessage,
  AppNotification,
} from '@/lib/types';
import {
  mockMessages as initialMessages,
} from '@/lib/mock-data';

interface ConnectionsProps {
  connections: ConnectionRequest[];
  setConnections: React.Dispatch<React.SetStateAction<ConnectionRequest[]>>;
  notifications: AppNotification[];
  setNotifications: React.Dispatch<React.SetStateAction<AppNotification[]>>;
}

// ─── Constants ───
const CURRENT_ORG_ID = 'org-exp-001';
const CURRENT_ORG_NAME = 'Your Company';

// ─── Status Config ───
const statusConfig: Record<
  ConnectionStatus,
  { label: string; color: string; icon: typeof Clock }
> = {
  pending: {
    label: 'Pending',
    color: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
    icon: Clock,
  },
  accepted: {
    label: 'Accepted',
    color: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
    icon: CheckCircle2,
  },
  declined: {
    label: 'Declined',
    color: 'bg-red-500/15 text-red-400 border-red-500/30',
    icon: XCircle,
  },
};

// ─── Time helpers ───
function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function formatTime(dateStr: string): string {
  return new Date(dateStr).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ─── ConnectionCard ───
function ConnectionCard({
  connection,
  unreadCount,
  isActive,
  onSelect,
}: {
  connection: ConnectionRequest;
  unreadCount: number;
  isActive: boolean;
  onSelect: () => void;
}) {
  const cfg = statusConfig[connection.status];
  const StatusIcon = cfg.icon;

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left p-4 transition-all border-b border-slate-700/30 hover:bg-slate-800/60 ${
        isActive ? 'bg-slate-800/80 border-l-2 border-l-sky-500' : ''
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1">
            <Building2 className="w-4 h-4 text-sky-400 flex-shrink-0" />
            <p className="text-sm font-semibold text-white truncate">
              {connection.importerOrgName}
            </p>
          </div>
          <div className="flex items-center gap-2 mb-2">
            <Globe className="w-3 h-3 text-slate-500" />
            <span className="text-xs text-slate-400">
              {connection.importerCountry}
            </span>
            <span className="text-xs text-slate-600">|</span>
            <span className="text-xs text-slate-400">
              {connection.importerIndustry}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className={`text-[10px] font-semibold px-2 py-0.5 border ${cfg.color}`}
            >
              <StatusIcon className="w-3 h-3 mr-1" />
              {cfg.label}
            </Badge>
            <span className="text-[10px] text-slate-500">
              {timeAgo(connection.createdAt)}
            </span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2 flex-shrink-0">
          <div className="bg-sky-500/15 border border-sky-500/30 rounded-md px-2 py-1">
            <span className="text-xs font-bold text-sky-400">
              {connection.finalScore}
            </span>
          </div>
          {unreadCount > 0 && (
            <div className="bg-sky-500 text-white text-[10px] font-bold rounded-full w-5 h-5 flex items-center justify-center">
              {unreadCount}
            </div>
          )}
        </div>
      </div>
    </button>
  );
}

// ─── Chat Bubble ───
function ChatBubble({ message }: { message: ChatMessage }) {
  const isOwn = message.senderRole === 'exporter';

  return (
    <div className={`flex ${isOwn ? 'justify-end' : 'justify-start'} mb-3`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-2.5 ${
          isOwn
            ? 'bg-sky-600 text-white rounded-br-md'
            : 'bg-slate-700/60 text-slate-200 border border-slate-600/30 rounded-bl-md'
        }`}
      >
        {!isOwn && (
          <p className="text-[10px] font-semibold text-sky-400 mb-1">
            {message.senderName}
          </p>
        )}
        <p className="text-sm leading-relaxed">{message.content}</p>
        <div
          className={`flex items-center gap-1 mt-1 ${
            isOwn ? 'justify-end' : 'justify-start'
          }`}
        >
          <span
            className={`text-[10px] ${
              isOwn ? 'text-sky-200/60' : 'text-slate-500'
            }`}
          >
            {formatTime(message.createdAt)}
          </span>
          {isOwn && message.read && (
            <CheckCircle2 className="w-3 h-3 text-sky-200/60" />
          )}
        </div>
      </div>
    </div>
  );
}

// ─── BuyerSimulationPanel ───
// Simulates the buyer side: accept/decline incoming requests
function BuyerSimulationPanel({
  connections,
  onRespond,
}: {
  connections: ConnectionRequest[];
  onRespond: (connectionId: string, action: 'accepted' | 'declined') => void;
}) {
  const pendingRequests = connections.filter((c) => c.status === 'pending');

  if (pendingRequests.length === 0) {
    return (
      <div className="text-center py-8">
        <Lock className="w-8 h-8 text-slate-600 mx-auto mb-3" />
        <p className="text-sm text-slate-400">No pending requests</p>
        <p className="text-xs text-slate-500 mt-1">
          When you connect with a match, the buyer will see it here.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {pendingRequests.map((conn) => (
        <Card
          key={conn.id}
          className="bg-slate-800/60 border-slate-700/50 p-4"
        >
          <div className="flex items-start gap-3 mb-3">
            <div className="w-10 h-10 rounded-full bg-sky-500/15 border border-sky-500/30 flex items-center justify-center flex-shrink-0">
              <UserPlus className="w-5 h-5 text-sky-400" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-white">
                {conn.exporterOrgName} wants to connect
              </p>
              <p className="text-xs text-slate-400 mt-0.5">
                Match Score: {conn.finalScore} | {conn.importerIndustry}
              </p>
              {conn.note && (
                <p className="text-xs text-slate-300 mt-2 bg-slate-700/30 rounded-lg p-2 border border-slate-600/20 leading-relaxed">
                  {`"${conn.note}"`}
                </p>
              )}
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={() => onRespond(conn.id, 'accepted')}
              size="sm"
              className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold"
            >
              <CheckCircle2 className="w-3.5 h-3.5 mr-1.5" />
              Accept
            </Button>
            <Button
              onClick={() => onRespond(conn.id, 'declined')}
              size="sm"
              variant="outline"
              className="flex-1 border-slate-600 text-slate-300 hover:bg-red-500/10 hover:text-red-400 hover:border-red-500/30 text-xs"
            >
              <XCircle className="w-3.5 h-3.5 mr-1.5" />
              Decline
            </Button>
          </div>
        </Card>
      ))}
    </div>
  );
}

// ─── Main Connections Component ───
export default function Connections({
  connections,
  setConnections,
  notifications,
  setNotifications,
}: ConnectionsProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);
  const [newMessage, setNewMessage] = useState('');
  const [filterStatus, setFilterStatus] = useState<ConnectionStatus | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'exporter' | 'buyer'>('exporter');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const selectedConnection = connections.find(
    (c) => c.id === selectedConnectionId
  );

  const connectionMessages = messages.filter(
    (m) => m.connectionId === selectedConnectionId
  );

  const unreadCountForConnection = useCallback(
    (connectionId: string) =>
      messages.filter(
        (m) =>
          m.connectionId === connectionId &&
          !m.read &&
          m.senderId !== CURRENT_ORG_ID
      ).length,
    [messages]
  );

  const totalUnread = notifications.filter((n) => !n.read).length;

  // Filter connections
  const filteredConnections = connections
    .filter((c) => filterStatus === 'all' || c.status === filterStatus)
    .filter(
      (c) =>
        !searchQuery ||
        c.importerOrgName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        c.importerCountry.toLowerCase().includes(searchQuery.toLowerCase())
    );

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [connectionMessages.length]);

  // Mark messages as read when selecting a connection
  useEffect(() => {
    if (!selectedConnectionId) return;
    setMessages((prev) =>
      prev.map((m) =>
        m.connectionId === selectedConnectionId && m.senderId !== CURRENT_ORG_ID
          ? { ...m, read: true }
          : m
      )
    );
    setNotifications((prev) =>
      prev.map((n) =>
        n.connectionId === selectedConnectionId ? { ...n, read: true } : n
      )
    );
  }, [selectedConnectionId]);

  // Send message
  const handleSendMessage = () => {
    if (!newMessage.trim() || !selectedConnection) return;
    if (selectedConnection.status !== 'accepted') return;

    const msg: ChatMessage = {
      id: `msg-${Date.now()}`,
      connectionId: selectedConnection.id,
      senderId: CURRENT_ORG_ID,
      senderName: CURRENT_ORG_NAME,
      senderRole: 'exporter',
      content: newMessage.trim(),
      createdAt: new Date().toISOString(),
      read: false,
    };

    setMessages((prev) => [...prev, msg]);
    setNewMessage('');

    // Simulate buyer reply after 2s
    setTimeout(() => {
      const reply: ChatMessage = {
        id: `msg-${Date.now()}-reply`,
        connectionId: selectedConnection.id,
        senderId: selectedConnection.importerOrgId,
        senderName: selectedConnection.importerOrgName,
        senderRole: 'importer',
        content: getSimulatedReply(),
        createdAt: new Date().toISOString(),
        read: false,
      };
      setMessages((prev) => [...prev, reply]);

      const notif: AppNotification = {
        id: `notif-${Date.now()}`,
        type: 'new_message',
        title: 'New Message',
        description: `${selectedConnection.importerOrgName} sent you a message.`,
        connectionId: selectedConnection.id,
        read: false,
        createdAt: new Date().toISOString(),
      };
      setNotifications((prev) => [notif, ...prev]);
    }, 2000);
  };

  // Buyer simulation: respond to connection
  const handleBuyerRespond = (
    connectionId: string,
    action: 'accepted' | 'declined'
  ) => {
    setConnections((prev) =>
      prev.map((c) =>
        c.id === connectionId
          ? { ...c, status: action, respondedAt: new Date().toISOString() }
          : c
      )
    );

    const conn = connections.find((c) => c.id === connectionId);
    if (!conn) return;

    const notif: AppNotification = {
      id: `notif-${Date.now()}`,
      type: action === 'accepted' ? 'connection_accepted' : 'connection_declined',
      title:
        action === 'accepted'
          ? 'Connection Accepted'
          : 'Connection Declined',
      description:
        action === 'accepted'
          ? `${conn.importerOrgName} accepted your connection request.`
          : `${conn.importerOrgName} declined your connection request.`,
      connectionId,
      read: false,
      createdAt: new Date().toISOString(),
    };
    setNotifications((prev) => [notif, ...prev]);

    // If accepted, add a welcome message from buyer
    if (action === 'accepted') {
      const welcomeMsg: ChatMessage = {
        id: `msg-${Date.now()}-welcome`,
        connectionId,
        senderId: conn.importerOrgId,
        senderName: conn.importerOrgName,
        senderRole: 'importer',
        content: `Hello! Thanks for connecting. We're interested in exploring potential partnerships in ${conn.importerIndustry}. How can we help each other?`,
        createdAt: new Date().toISOString(),
        read: false,
      };
      setMessages((prev) => [...prev, welcomeMsg]);
    }
  };

  // Key handler for sending
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="space-y-4">
      {/* View toggle (Exporter / Buyer sim) */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-white">Connections</h2>
        <div className="flex items-center gap-2">
          {totalUnread > 0 && (
            <Badge
              variant="outline"
              className="bg-sky-500/15 border-sky-500/30 text-sky-400 text-xs"
            >
              {totalUnread} unread
            </Badge>
          )}
          <div className="flex bg-slate-800/60 rounded-lg border border-slate-700/50 p-0.5">
            <button
              onClick={() => setViewMode('exporter')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                viewMode === 'exporter'
                  ? 'bg-sky-600 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Exporter View
            </button>
            <button
              onClick={() => setViewMode('buyer')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                viewMode === 'buyer'
                  ? 'bg-sky-600 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Buyer Sim
            </button>
          </div>
        </div>
      </div>

      {/* ─── Buyer Simulation ─── */}
      {viewMode === 'buyer' && (
        <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-5">
          <div className="flex items-center gap-2 mb-4">
            <UserPlus className="w-5 h-5 text-sky-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Incoming Connection Requests
            </h3>
            <Badge
              variant="outline"
              className="bg-slate-700/30 border-slate-600/50 text-slate-400 text-[10px]"
            >
              Buyer Side
            </Badge>
          </div>
          <p className="text-xs text-slate-400 mb-4 leading-relaxed">
            This simulates what the buyer sees when you send a connection
            request. Accept to unlock chat, or decline.
          </p>
          <BuyerSimulationPanel
            connections={connections}
            onRespond={handleBuyerRespond}
          />
        </Card>
      )}

      {/* ─── Exporter View: Connection List + Chat ─── */}
      {viewMode === 'exporter' && (
        <div className="flex gap-0 h-[600px] bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl border border-slate-700/50 overflow-hidden">
          {/* Left Panel: Connection List */}
          <div
            className={`w-full sm:w-80 flex-shrink-0 border-r border-slate-700/40 flex flex-col ${
              selectedConnectionId ? 'hidden sm:flex' : 'flex'
            }`}
          >
            {/* Search & Filters */}
            <div className="p-3 border-b border-slate-700/40 space-y-2">
              <div className="flex items-center gap-2 bg-slate-700/30 rounded-lg px-3 py-2 border border-slate-600/30">
                <Search className="w-4 h-4 text-slate-500" />
                <input
                  type="text"
                  placeholder="Search connections..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-transparent text-sm text-white placeholder-slate-500 outline-none flex-1"
                />
              </div>
              <div className="flex gap-1.5">
                {(['all', 'accepted', 'pending', 'declined'] as const).map(
                  (s) => (
                    <button
                      key={s}
                      onClick={() => setFilterStatus(s)}
                      className={`px-2.5 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wider transition-all ${
                        filterStatus === s
                          ? 'bg-sky-600/20 text-sky-400 border border-sky-500/30'
                          : 'text-slate-500 hover:text-slate-300 border border-transparent'
                      }`}
                    >
                      {s}
                    </button>
                  )
                )}
              </div>
            </div>

            {/* Connection Items */}
            <ScrollArea className="flex-1">
              {filteredConnections.length === 0 ? (
                <div className="text-center py-12 px-4">
                  <MessageSquare className="w-8 h-8 text-slate-600 mx-auto mb-3" />
                  <p className="text-sm text-slate-400">No connections found</p>
                  <p className="text-xs text-slate-500 mt-1">
                    Connect with matches to start conversations.
                  </p>
                </div>
              ) : (
                filteredConnections.map((conn) => (
                  <ConnectionCard
                    key={conn.id}
                    connection={conn}
                    unreadCount={unreadCountForConnection(conn.id)}
                    isActive={selectedConnectionId === conn.id}
                    onSelect={() => setSelectedConnectionId(conn.id)}
                  />
                ))
              )}
            </ScrollArea>
          </div>

          {/* Right Panel: Chat Area */}
          <div
            className={`flex-1 flex flex-col ${
              !selectedConnectionId ? 'hidden sm:flex' : 'flex'
            }`}
          >
            {!selectedConnection ? (
              // Empty state
              <div className="flex-1 flex flex-col items-center justify-center text-center p-6">
                <div className="w-16 h-16 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center mb-4">
                  <MessageSquare className="w-7 h-7 text-slate-500" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">
                  Select a connection
                </h3>
                <p className="text-sm text-slate-400 max-w-xs leading-relaxed">
                  Choose a connection from the list to view status or start
                  chatting.
                </p>
              </div>
            ) : (
              <>
                {/* Chat Header */}
                <div className="flex items-center gap-3 p-4 border-b border-slate-700/40">
                  <button
                    onClick={() => setSelectedConnectionId(null)}
                    className="sm:hidden text-slate-400 hover:text-white"
                  >
                    <ArrowLeft className="w-5 h-5" />
                  </button>
                  <div className="w-10 h-10 rounded-full bg-sky-500/15 border border-sky-500/30 flex items-center justify-center flex-shrink-0">
                    <Building2 className="w-5 h-5 text-sky-400" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-white truncate">
                      {selectedConnection.importerOrgName}
                    </p>
                    <div className="flex items-center gap-2">
                      <Badge
                        variant="outline"
                        className={`text-[10px] font-semibold px-1.5 py-0 border ${
                          statusConfig[selectedConnection.status].color
                        }`}
                      >
                        {statusConfig[selectedConnection.status].label}
                      </Badge>
                      <span className="text-[10px] text-slate-500">
                        {selectedConnection.importerCountry} |{' '}
                        {selectedConnection.importerIndustry}
                      </span>
                    </div>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="bg-sky-500/15 border border-sky-500/30 rounded-md px-2.5 py-1">
                          <span className="text-xs font-bold text-sky-400">
                            {selectedConnection.finalScore}
                          </span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="bg-slate-800 border-slate-700 text-slate-200">
                        <p className="text-xs">Match Score</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>

                {/* Chat Messages / Status Area */}
                <ScrollArea className="flex-1 p-4">
                  {selectedConnection.status === 'pending' && (
                    <div className="flex flex-col items-center justify-center h-full text-center py-12">
                      <div className="w-14 h-14 rounded-full bg-amber-500/15 border border-amber-500/30 flex items-center justify-center mb-4">
                        <Clock className="w-6 h-6 text-amber-400" />
                      </div>
                      <h3 className="text-base font-bold text-white mb-2">
                        Connection Pending
                      </h3>
                      <p className="text-sm text-slate-400 max-w-xs leading-relaxed mb-2">
                        Your connection request has been sent to{' '}
                        <span className="text-white font-semibold">
                          {selectedConnection.importerOrgName}
                        </span>
                        . Chat will unlock once they accept.
                      </p>
                      {selectedConnection.note && (
                        <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600/20 mt-3 max-w-xs">
                          <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-1">
                            Your note
                          </p>
                          <p className="text-xs text-slate-300 leading-relaxed">
                            {selectedConnection.note}
                          </p>
                        </div>
                      )}
                      <p className="text-xs text-slate-500 mt-4">
                        Tip: Switch to{' '}
                        <span className="text-sky-400 font-semibold">
                          Buyer Sim
                        </span>{' '}
                        to simulate accepting this request.
                      </p>
                    </div>
                  )}

                  {selectedConnection.status === 'declined' && (
                    <div className="flex flex-col items-center justify-center h-full text-center py-12">
                      <div className="w-14 h-14 rounded-full bg-red-500/15 border border-red-500/30 flex items-center justify-center mb-4">
                        <XCircle className="w-6 h-6 text-red-400" />
                      </div>
                      <h3 className="text-base font-bold text-white mb-2">
                        Connection Declined
                      </h3>
                      <p className="text-sm text-slate-400 max-w-xs leading-relaxed">
                        <span className="text-white font-semibold">
                          {selectedConnection.importerOrgName}
                        </span>{' '}
                        has declined your connection request. You can explore
                        other matches.
                      </p>
                    </div>
                  )}

                  {selectedConnection.status === 'accepted' && (
                    <div>
                      {/* Connected banner */}
                      <div className="flex items-center justify-center gap-2 py-3 mb-4">
                        <Separator className="flex-1 bg-slate-700/50" />
                        <Badge
                          variant="outline"
                          className="bg-emerald-500/10 border-emerald-500/30 text-emerald-400 text-[10px]"
                        >
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Connected{' '}
                          {selectedConnection.respondedAt &&
                            timeAgo(selectedConnection.respondedAt)}
                        </Badge>
                        <Separator className="flex-1 bg-slate-700/50" />
                      </div>

                      {/* Messages */}
                      {connectionMessages.length === 0 ? (
                        <div className="text-center py-8">
                          <p className="text-sm text-slate-400">
                            Connection established! Start the conversation.
                          </p>
                        </div>
                      ) : (
                        connectionMessages.map((msg) => (
                          <ChatBubble key={msg.id} message={msg} />
                        ))
                      )}
                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </ScrollArea>

                {/* Message Input */}
                {selectedConnection.status === 'accepted' ? (
                  <div className="p-3 border-t border-slate-700/40">
                    <div className="flex items-end gap-2">
                      <textarea
                        value={newMessage}
                        onChange={(e) => setNewMessage(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Type a message..."
                        rows={1}
                        className="flex-1 bg-slate-700/30 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 border border-slate-600/30 outline-none focus:border-sky-500/50 resize-none min-h-[40px] max-h-[100px]"
                      />
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              onClick={handleSendMessage}
                              size="icon"
                              disabled={!newMessage.trim()}
                              className="h-10 w-10 bg-sky-600 hover:bg-sky-700 text-white disabled:opacity-30 rounded-xl flex-shrink-0"
                            >
                              <Send className="w-4 h-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent className="bg-slate-800 border-slate-700 text-slate-200">
                            <p className="text-xs">
                              Send message (Enter)
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  </div>
                ) : (
                  <div className="p-3 border-t border-slate-700/40">
                    <div className="flex items-center justify-center gap-2 py-2 text-slate-500">
                      <Lock className="w-4 h-4" />
                      <span className="text-xs">
                        {selectedConnection.status === 'pending'
                          ? 'Chat locked until connection is accepted'
                          : 'Chat unavailable for declined connections'}
                      </span>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Simulated replies
function getSimulatedReply(): string {
  const replies = [
    "That sounds very interesting! Let me review the details with our procurement team and get back to you.",
    "We appreciate the quick response. Could you also share your certifications and compliance documentation?",
    "Excellent! Our team is reviewing your proposal. We'll schedule a call to discuss specifics soon.",
    "Thank you for the information. We're currently comparing a few suppliers and will make a decision by end of week.",
    "Great to hear! We have some specific requirements around packaging. Can you accommodate custom labeling?",
  ];
  return replies[Math.floor(Math.random() * replies.length)];
}
