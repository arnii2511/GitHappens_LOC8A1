'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import {
  ThumbsUp,
  ThumbsDown,
  MapPin,
  Package,
  TrendingUp,
  Calendar,
  Bookmark,
  Ban,
  Shield,
  Sparkles,
  Globe,
  Building2,
  Clock,
} from 'lucide-react';
import ScoreBreakdown from '@/components/ScoreBreakdown';
import type { MatchCardDTO, SwipeAction, ConnectionRequest, FeedResponse } from '@/lib/types';

interface MatchingCardsProps {
  onConnect?: (connection: ConnectionRequest) => void;
  onSwipeAction?: (payload: { action: SwipeAction; finalScore: number }) => void;
  accountType?: 'exporter' | 'buyer';
}

function createSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

// ─── Trust Score Badge ───
function TrustBadge({ score }: { score: number }) {
  const level = score >= 85 ? 'High' : score >= 70 ? 'Medium' : 'Low';
  const color =
    score >= 85
      ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
      : score >= 70
        ? 'bg-sky-500/20 text-sky-400 border-sky-500/30'
        : 'bg-amber-500/20 text-amber-400 border-amber-500/30';

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs font-semibold ${color}`}>
            <Shield className="w-3.5 h-3.5" />
            <span>Trust: {score}</span>
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="bg-slate-800 border-slate-700 text-slate-200">
          <p className="text-xs">Trust Level: {level} ({score}/100)</p>
          <p className="text-xs text-slate-400">Based on RV, TH, DF, PB scores</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// ─── Product Tags ───
function ProductTags({ products }: { products: MatchCardDTO['importerProducts'] }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {products.map((p) => (
        <TooltipProvider key={p.hsCode}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant="outline"
                className="bg-slate-700/40 border-slate-600/50 text-slate-300 text-[11px] font-medium hover:bg-slate-700/60 cursor-default"
              >
                {p.category}
              </Badge>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="bg-slate-800 border-slate-700 text-slate-200 max-w-xs">
              <p className="text-xs font-semibold">{p.category}</p>
              <p className="text-xs text-slate-400">HS Code: {p.hsCode}</p>
              <p className="text-xs text-slate-400">{p.description}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ))}
    </div>
  );
}

// ─── Reason Card ───
function ReasonCard({ reason, index }: { reason: MatchCardDTO['reasons'][0]; index: number }) {
  return (
    <div className="flex gap-3 items-start">
      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-sky-500/20 border border-sky-500/30 flex items-center justify-center">
        <span className="text-[10px] font-bold text-sky-400">{index + 1}</span>
      </div>
      <div className="min-w-0">
        <p className="text-sm font-semibold text-slate-200 leading-tight">{reason.label}</p>
        <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">{reason.description}</p>
      </div>
    </div>
  );
}

// ─── Main MatchingCards Component ───
export default function MatchingCards({
  onConnect,
  onSwipeAction,
  accountType = 'exporter',
}: MatchingCardsProps = {}) {
  const [feed, setFeed] = useState<MatchCardDTO[]>([]);
  const [swipedIds, setSwipedIds] = useState<Set<string>>(new Set());
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationDirection, setAnimationDirection] = useState<'left' | 'right' | 'up' | 'down' | null>(null);
  const [isLoadingFeed, setIsLoadingFeed] = useState(true);
  const [feedError, setFeedError] = useState<string | null>(null);
  const sessionIdRef = useRef<string>(createSessionId());
  const shownAtRef = useRef<Record<string, number>>({});

  const loadFeed = useCallback(async () => {
    setIsLoadingFeed(true);
    setFeedError(null);

    try {
      const params = new URLSearchParams({ limit: '50' });
      params.set('role', accountType);
      if (process.env.NEXT_PUBLIC_DEMO_BUYER_ID) {
        params.set('buyerId', process.env.NEXT_PUBLIC_DEMO_BUYER_ID);
      }

      const res = await fetch(`/api/matches/feed?${params.toString()}`, {
        method: 'GET',
        cache: 'no-store',
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Failed to load feed (${res.status})`);
      }

      const data = (await res.json()) as FeedResponse & { buyerId?: string };
      setFeed(data.cards || []);
      setSwipedIds(new Set());
      shownAtRef.current = {};
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load feed.';
      setFeedError(message);
    } finally {
      setIsLoadingFeed(false);
    }
  }, [accountType]);

  useEffect(() => {
    loadFeed();
  }, [loadFeed]);

  // Active cards = feed minus swiped
  const activeCards = feed.filter((c) => !swipedIds.has(c.matchId));
  const currentCard = activeCards.length > 0 ? activeCards[0] : null;
  const remainingCount = activeCards.length;

  useEffect(() => {
    if (!currentCard) return;
    if (!shownAtRef.current[currentCard.matchId]) {
      shownAtRef.current[currentCard.matchId] = Date.now();
    }
  }, [currentCard]);

  // Swipe handler (unified for all actions)
  const handleSwipe = useCallback(
    async (action: SwipeAction) => {
      if (!currentCard || isAnimating) return;

      const directionMap: Record<SwipeAction, 'left' | 'right' | 'up' | 'down'> = {
        like: 'right',
        dislike: 'left',
        save: 'up',
        block: 'down',
      };
      setAnimationDirection(directionMap[action]);
      setIsAnimating(true);

      // Call API endpoint
      try {
        const shownAt = shownAtRef.current[currentCard.matchId] || Date.now();
        const dwellMs = Math.max(0, Date.now() - shownAt);
        const shownRank = Number(currentCard.shownRank || 1) || 1;
        const source = currentCard.candidateSource || 'recommended';
        const recommendationVersion =
          currentCard.recommendationVersion ||
          process.env.NEXT_PUBLIC_RECOMMENDATION_VERSION ||
          'hybrid-v1';
        const actionPath = action === 'like' ? 'connect' : action === 'dislike' ? 'skip' : action;
        await fetch(`/api/matches/${currentCard.matchId}/${actionPath}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            matchId: currentCard.matchId,
            action,
            buyerId: currentCard.buyerId,
            exporterId: currentCard.exporterId || currentCard.matchId,
            session_id: sessionIdRef.current,
            shown_rank: shownRank,
            source,
            dwell_ms: dwellMs,
            recommendation_version: recommendationVersion,
          }),
        });
      } catch {
        // Swipe still works locally even if API fails (optimistic)
      }

      // Create pending connection request on Connect
      if (action === 'like' && onConnect) {
        const newConnection: ConnectionRequest = {
          id: `conn-${Date.now()}`,
          matchId: currentCard.matchId,
          exporterOrgId: 'org-exp-001',
          exporterOrgName: 'Your Company',
          importerOrgId: currentCard.importerOrg.id,
          importerOrgName: currentCard.importerOrg.name,
          importerCountry: currentCard.importerOrg.country,
          importerIndustry: currentCard.importerOrg.industry,
          finalScore: currentCard.finalScore,
          status: 'pending',
          createdAt: new Date().toISOString(),
          note: `Interested in partnering for ${currentCard.importerProducts.map(p => p.category).join(', ')}.`,
        };
        onConnect(newConnection);
      }
      onSwipeAction?.({ action, finalScore: currentCard.finalScore });

      // Remove card after animation
      setTimeout(() => {
        setSwipedIds((prev) => new Set([...prev, currentCard.matchId]));
        setIsAnimating(false);
        setAnimationDirection(null);
      }, 300);
    },
    [currentCard, isAnimating, onConnect, onSwipeAction]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') handleSwipe('dislike');
      if (e.key === 'ArrowRight') handleSwipe('like');
      if (e.key === 'ArrowUp') { e.preventDefault(); handleSwipe('save'); }
      if (e.key === 'ArrowDown') { e.preventDefault(); handleSwipe('block'); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [handleSwipe]);

  // Animation class
  const getAnimationClass = () => {
    if (!isAnimating || !animationDirection) return 'translate-x-0 opacity-100';
    const map = {
      left: '-translate-x-full opacity-0',
      right: 'translate-x-full opacity-0',
      up: '-translate-y-32 opacity-0',
      down: 'translate-y-32 opacity-0',
    };
    return map[animationDirection];
  };

  // Score color gradient
  const getScoreGradient = (score: number) =>
    score >= 90
      ? 'from-emerald-500 to-emerald-600'
      : score >= 80
        ? 'from-sky-500 to-sky-600'
        : score >= 70
          ? 'from-amber-500 to-amber-600'
          : 'from-red-500 to-red-600';

  // ─── Empty state ───
  if (isLoadingFeed) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <h3 className="text-xl font-bold text-white mb-2">Loading matches...</h3>
        <p className="text-slate-400 max-w-sm leading-relaxed">
          Fetching ranked opportunities from the ML backend.
        </p>
      </div>
    );
  }

  if (feedError) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <h3 className="text-xl font-bold text-white mb-2">Could not load matches</h3>
        <p className="text-slate-400 max-w-sm leading-relaxed">{feedError}</p>
        <Button onClick={loadFeed} className="mt-6 bg-sky-600 hover:bg-sky-700 text-white">
          Retry
        </Button>
      </div>
    );
  }

  if (!currentCard) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <div className="w-20 h-20 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center mb-6">
          <Sparkles className="w-8 h-8 text-slate-500" />
        </div>
        <h3 className="text-xl font-bold text-white mb-2">No more matches</h3>
        <p className="text-slate-400 max-w-sm leading-relaxed">
          {"You've reviewed all current matches. New matches will appear as our algorithm processes fresh market signals."}
        </p>
        <Button
          onClick={() => {
            if (feed.length === 0) {
              void loadFeed();
            } else {
              setSwipedIds(new Set());
            }
          }}
          className="mt-6 bg-sky-600 hover:bg-sky-700 text-white"
        >
          Reset Feed (Demo)
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Feed indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-bold text-white">Match Feed</h2>
          <Badge variant="outline" className="bg-slate-800 border-slate-700 text-slate-300 text-xs">
            {remainingCount} remaining
          </Badge>
        </div>
        <div className="hidden sm:flex items-center gap-2 text-xs text-slate-500">
          <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono">{'<-'}</kbd>
          <span>Skip</span>
          <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono">{'->'}</kbd>
          <span>Connect</span>
          <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono">{'Up'}</kbd>
          <span>Save</span>
          <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono">{'Dn'}</kbd>
          <span>Block</span>
        </div>
      </div>

      {/* Main layout: Card + Breakdown side-by-side */}
      <div className="flex gap-6 items-start">
        {/* Card Stack */}
        <div className="flex-1 min-w-0">
          <div className="relative">
            {/* Stacked card shadows behind */}
            {activeCards.length > 2 && (
              <div className="absolute inset-x-4 top-3 h-full bg-slate-800/40 rounded-2xl border border-slate-700/30" />
            )}
            {activeCards.length > 1 && (
              <div className="absolute inset-x-2 top-1.5 h-full bg-slate-800/60 rounded-2xl border border-slate-700/40" />
            )}

            {/* Active Card */}
            <div className={`relative transition-all duration-300 ease-out ${getAnimationClass()}`}>
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl border border-slate-700/50 overflow-hidden shadow-2xl hover:shadow-sky-500/10 transition-shadow duration-300">
                <div className="p-6 space-y-5">
                  {/* Header */}
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0 space-y-1.5">
                      <div className="flex items-center gap-2">
                        <Building2 className="w-5 h-5 text-sky-400 flex-shrink-0" />
                        <h2 className="text-2xl font-bold text-white truncate">{currentCard.importerOrg.name}</h2>
                      </div>
                      <div className="flex items-center gap-3 flex-wrap">
                        <div className="flex items-center gap-1.5 text-slate-400 text-sm">
                          <Globe className="w-3.5 h-3.5" />
                          <span>{currentCard.importerOrg.country}</span>
                        </div>
                        <Badge variant="outline" className="bg-slate-700/30 border-slate-600/50 text-slate-300 text-[11px]">
                          {currentCard.importerOrg.companySize}
                        </Badge>
                        <Badge variant="outline" className="bg-slate-700/30 border-slate-600/50 text-slate-300 text-[11px]">
                          {currentCard.importerOrg.industry}
                        </Badge>
                      </div>
                    </div>

                    {/* Scores */}
                    <div className="flex items-center gap-3 flex-shrink-0">
                      <TrustBadge score={currentCard.trustScore} />
                      <div className={`bg-gradient-to-br ${getScoreGradient(currentCard.finalScore)} rounded-xl px-4 py-2.5 shadow-lg`}>
                        <p className="text-2xl font-bold text-white leading-none">{currentCard.finalScore}</p>
                        <p className="text-[10px] text-white/70 uppercase tracking-wider mt-0.5">Final</p>
                      </div>
                    </div>
                  </div>

                  {/* Details Grid */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-slate-700/20 rounded-xl p-4 border border-slate-600/30">
                      <div className="flex items-center gap-2 mb-2">
                        <MapPin className="w-4 h-4 text-sky-400" />
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Location</p>
                      </div>
                      <p className="text-sm font-semibold text-white">
                        {currentCard.importerLocation
                          ? `${currentCard.importerLocation.city}, ${currentCard.importerLocation.country}`
                          : currentCard.importerOrg.country}
                      </p>
                      {currentCard.importerLocation?.region && (
                        <p className="text-xs text-slate-400 mt-0.5">{currentCard.importerLocation.region}</p>
                      )}
                    </div>

                    <div className="bg-slate-700/20 rounded-xl p-4 border border-slate-600/30">
                      <div className="flex items-center gap-2 mb-2">
                        <TrendingUp className="w-4 h-4 text-sky-400" />
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Volume</p>
                      </div>
                      <p className="text-sm font-semibold text-white">{currentCard.quantity}</p>
                      <p className="text-xs text-slate-400 mt-0.5">Monthly demand</p>
                    </div>

                    <div className="bg-slate-700/20 rounded-xl p-4 border border-slate-600/30">
                      <div className="flex items-center gap-2 mb-2">
                        <Calendar className="w-4 h-4 text-sky-400" />
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Timeline</p>
                      </div>
                      <p className="text-sm font-semibold text-white">{currentCard.timeline}</p>
                      <div className="flex items-center gap-1 mt-0.5">
                        <Clock className="w-3 h-3 text-slate-500" />
                        <p className="text-xs text-slate-400">Active {currentCard.lastActive}</p>
                      </div>
                    </div>

                    <div className="bg-slate-700/20 rounded-xl p-4 border border-slate-600/30">
                      <div className="flex items-center gap-2 mb-2">
                        <Package className="w-4 h-4 text-sky-400" />
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Products</p>
                      </div>
                      <ProductTags products={currentCard.importerProducts} />
                    </div>
                  </div>

                  {/* Top 3 "Why this match?" Reasons */}
                  <div className="bg-slate-700/10 rounded-xl p-4 border border-slate-600/20">
                    <div className="flex items-center gap-2 mb-3">
                      <Sparkles className="w-4 h-4 text-sky-400" />
                      <p className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Why this match?</p>
                    </div>
                    <div className="space-y-3">
                      {currentCard.reasons.slice(0, 3).map((reason, i) => (
                        <ReasonCard key={reason.rank} reason={reason} index={i} />
                      ))}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-3 pt-2">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            onClick={() => handleSwipe('block')}
                            variant="outline"
                            size="icon"
                            className="h-12 w-12 border-slate-600 text-slate-400 hover:bg-red-500/10 hover:text-red-400 hover:border-red-500/30"
                            disabled={isAnimating}
                          >
                            <Ban className="w-5 h-5" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent className="bg-slate-800 border-slate-700 text-slate-200">
                          <p className="text-xs">Block this match permanently</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>

                    <Button
                      onClick={() => handleSwipe('dislike')}
                      variant="outline"
                      className="flex-1 h-12 border-slate-600 text-slate-300 hover:bg-slate-700/50 hover:text-slate-100 hover:border-slate-500"
                      disabled={isAnimating}
                    >
                      <ThumbsDown className="w-5 h-5 mr-2" />
                      Skip
                    </Button>

                    <Button
                      onClick={() => handleSwipe('like')}
                      className="flex-1 h-12 bg-sky-600 hover:bg-sky-700 text-white font-semibold shadow-lg shadow-sky-600/20"
                      disabled={isAnimating}
                    >
                      <ThumbsUp className="w-5 h-5 mr-2" />
                      Connect
                    </Button>

                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            onClick={() => handleSwipe('save')}
                            variant="outline"
                            size="icon"
                            className="h-12 w-12 border-slate-600 text-slate-400 hover:bg-amber-500/10 hover:text-amber-400 hover:border-amber-500/30"
                            disabled={isAnimating}
                          >
                            <Bookmark className="w-5 h-5" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent className="bg-slate-800 border-slate-700 text-slate-200">
                          <p className="text-xs">Save for later</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>

                  {/* Card position indicator */}
                  <div className="flex items-center justify-center gap-1.5">
                    {activeCards.slice(0, 6).map((card, i) => (
                      <div
                        key={card.matchId}
                        className={`h-1.5 rounded-full transition-all ${
                          i === 0 ? 'w-6 bg-sky-500' : 'w-1.5 bg-slate-600'
                        }`}
                      />
                    ))}
                    {activeCards.length > 6 && (
                      <span className="text-[10px] text-slate-500 ml-1">+{activeCards.length - 6}</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Score Breakdown Panel (desktop) */}
        <div className="w-80 flex-shrink-0 hidden lg:block">
          <ScoreBreakdown
            factors={currentCard.factors}
            trustComponents={currentCard.trustComponents}
            reasons={currentCard.reasons}
            matchScore={currentCard.matchScore}
            trustScore={currentCard.trustScore}
            finalScore={currentCard.finalScore}
          />
        </div>
      </div>

      {/* Mobile score breakdown */}
      <div className="lg:hidden">
        <ScoreBreakdown
          factors={currentCard.factors}
          trustComponents={currentCard.trustComponents}
          reasons={currentCard.reasons}
          matchScore={currentCard.matchScore}
          trustScore={currentCard.trustScore}
          finalScore={currentCard.finalScore}
        />
      </div>
    </div>
  );
}
