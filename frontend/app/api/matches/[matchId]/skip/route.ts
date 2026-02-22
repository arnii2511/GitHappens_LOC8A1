import { NextRequest, NextResponse } from 'next/server';
import type { SwipeResponse } from '@/lib/types';

const ML_API_BASE_URL =
  process.env.ML_API_BASE_URL ||
  process.env.NEXT_PUBLIC_ML_API_BASE_URL ||
  'http://127.0.0.1:8000';

type SwipeBridgeBody = {
  buyerId?: string;
  buyer_id?: string;
  exporterId?: string;
  exporter_id?: string;
  sessionId?: string;
  session_id?: string;
  shownRank?: number;
  shown_rank?: number;
  source?: string;
  dwellMs?: number;
  dwell_ms?: number;
  recommendationVersion?: string;
  recommendation_version?: string;
  device?: string;
  region?: string;
};

/**
 * POST /api/matches/:matchId/skip
 *
 * Records a 'dislike' swipe.
 *
 * In production this would:
 *  - Insert into swipes(user_id, match_id, action='dislike')
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ matchId: string }> }
) {
  const { matchId } = await params;
  const body = (await request.json().catch(() => ({}))) as SwipeBridgeBody;

  const buyerId = String(body.buyerId || body.buyer_id || '').trim();
  const exporterId = String(body.exporterId || body.exporter_id || matchId).trim();
  const sessionId = String(body.sessionId || body.session_id || '').trim();
  const shownRankRaw = Number(body.shownRank ?? body.shown_rank ?? NaN);
  const shownRank = Number.isFinite(shownRankRaw) && shownRankRaw > 0 ? Math.floor(shownRankRaw) : undefined;
  const source = String(body.source || 'recommended').trim().toLowerCase();
  const dwellRaw = Number(body.dwellMs ?? body.dwell_ms ?? NaN);
  const dwellMs = Number.isFinite(dwellRaw) && dwellRaw >= 0 ? Math.floor(dwellRaw) : undefined;
  const recommendationVersion = String(
    body.recommendationVersion || body.recommendation_version || process.env.RECOMMENDATION_VERSION || 'hybrid-v1'
  ).trim();

  const ua = request.headers.get('user-agent') || '';
  const inferredDevice = /mobile/i.test(ua) ? 'mobile' : /tablet|ipad/i.test(ua) ? 'tablet' : 'desktop';
  const regionHeader = request.headers.get('x-vercel-ip-country') || request.headers.get('x-country-code') || '';
  const device = String(body.device || inferredDevice).trim().toLowerCase();
  const region = String(body.region || regionHeader || '').trim().toUpperCase();

  if (!buyerId || !exporterId) {
    return NextResponse.json(
      { success: false, error: 'buyerId and exporterId are required for swipe forwarding.' },
      { status: 400 }
    );
  }

  const upstream = await fetch(`${ML_API_BASE_URL}/swipe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      buyer_id: buyerId,
      exporter_id: exporterId,
      action: 'left',
      session_id: sessionId || undefined,
      shown_rank: shownRank,
      source: source || 'recommended',
      dwell_ms: dwellMs,
      device: device || undefined,
      region: region || undefined,
      recommendation_version: recommendationVersion,
    }),
    cache: 'no-store',
  });

  if (!upstream.ok) {
    const details = await upstream.text();
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to persist swipe in ML backend.',
        details: details || upstream.statusText,
      },
      { status: 502 }
    );
  }

  const response: SwipeResponse = {
    success: true,
    action: 'dislike',
    matchId,
  };

  return NextResponse.json(response);
}
