'use client';

import Link from 'next/link';
import { Globe, Zap, Bell, UserCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface HeaderProps {
  notificationCount?: number;
  onNotificationClick?: () => void;
}

export default function Header({ notificationCount = 0, onNotificationClick }: HeaderProps) {
  return (
    <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-cyan-500 rounded-lg blur opacity-50"></div>
            <div className="relative bg-slate-900 px-3 py-2 rounded-lg flex items-center gap-2">
              <Globe className="w-6 h-6 text-cyan-400" />
              <span className="font-bold text-lg bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">NexPort</span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm">
            <Zap className="w-4 h-4 text-yellow-500" />
            <span className="hidden sm:inline">AI-Powered Trade Matching</span>
          </div>

          {/* Profile Link */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Link
                  href="/profile"
                  className="relative p-2 rounded-lg hover:bg-slate-800 transition-colors"
                  aria-label="View profile"
                >
                  <UserCircle className="w-5 h-5 text-slate-400" />
                </Link>
              </TooltipTrigger>
              <TooltipContent className="bg-slate-800 border-slate-700 text-slate-200">
                <p className="text-xs">View your profile</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          {/* Notification Bell */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={onNotificationClick}
                  className="relative p-2 rounded-lg hover:bg-slate-800 transition-colors"
                  aria-label={`Notifications${notificationCount > 0 ? `, ${notificationCount} unread` : ''}`}
                >
                  <Bell className="w-5 h-5 text-slate-400" />
                  {notificationCount > 0 && (
                    <Badge
                      className="absolute -top-0.5 -right-0.5 bg-sky-500 text-white text-[9px] font-bold px-1.5 py-0 min-w-[18px] h-[18px] flex items-center justify-center rounded-full border-2 border-slate-900"
                    >
                      {notificationCount > 9 ? '9+' : notificationCount}
                    </Badge>
                  )}
                </button>
              </TooltipTrigger>
              <TooltipContent className="bg-slate-800 border-slate-700 text-slate-200">
                <p className="text-xs">
                  {notificationCount > 0
                    ? `${notificationCount} unread notification${notificationCount > 1 ? 's' : ''}`
                    : 'No new notifications'}
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
    </header>
  );
}
