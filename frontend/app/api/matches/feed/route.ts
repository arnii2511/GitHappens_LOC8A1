import { NextRequest, NextResponse } from 'next/server';
import type { FeedResponse, MatchCardDTO } from '@/lib/types';

const ML_API_BASE_URL =
  process.env.ML_API_BASE_URL ||
  process.env.NEXT_PUBLIC_ML_API_BASE_URL ||
  'http://127.0.0.1:8000';

const DEFAULT_BUYER_ID =
  process.env.DEFAULT_BUYER_ID ||
  process.env.NEXT_PUBLIC_DEMO_BUYER_ID ||
  '';

type BackendFeedCard = {
  buyer_id?: string;
  exporter_id?: string;
  exporter_state?: string;
  exporter_cert?: string;
  match_score?: number;
  trust_score?: number;
  ml_score?: number;
  collab_score?: number;
  ltr_score?: number;
  final_rank?: number;
  retrieval_score?: number;
  text_similarity?: number;
  industry_similarity?: number;
  industry_assoc_score?: number;
  confidence?: number;
  reasons?: string[];
  warning?: string | null;
};

function toScore(value: unknown, fallback = 0): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(100, n));
}

function reasonLabel(raw: string, idx: number): { label: string; description: string } {
  const value = String(raw || '').trim();
  if (!value) return { label: `Reason ${idx + 1}`, description: 'Model-ranked candidate.' };

  const splitAt = value.indexOf(':');
  if (splitAt === -1) return { label: `Reason ${idx + 1}`, description: value };

  const label = value.slice(0, splitAt).trim() || `Reason ${idx + 1}`;
  const description = value.slice(splitAt + 1).trim() || value;
  return { label, description };
}

function mapBackendCard(card: BackendFeedCard, buyerId: string): MatchCardDTO {
  const exporterId = String(card.exporter_id || '');
  const matchScore = toScore(card.match_score, 0);
  const trustScore = toScore(card.trust_score, 0);
  const finalScore = toScore(card.final_rank, Math.max(matchScore, trustScore));

  const factors: MatchCardDTO['factors'] = {
    productFit: toScore(card.industry_similarity, matchScore),
    geographyFit: toScore(card.retrieval_score, 50),
    buyerActivity: toScore(card.ml_score, finalScore),
    scaleFit: toScore(card.collab_score, 50),
    marketTrend: toScore(card.text_similarity, 50),
    tradeFrequency: toScore(card.industry_assoc_score, 50),
  };

  const trustComponents: MatchCardDTO['trustComponents'] = {
    registrationVerification: trustScore,
    tradeHistory: trustScore,
    documentationFidelity: trustScore,
    paymentBehavior: trustScore,
  };

  const factorTypes: Array<keyof MatchCardDTO['factors']> = [
    'productFit',
    'geographyFit',
    'buyerActivity',
    'scaleFit',
    'marketTrend',
    'tradeFrequency',
  ];

  const rawReasons = Array.isArray(card.reasons) ? card.reasons : [];
  const reasons: MatchCardDTO['reasons'] =
    rawReasons.length > 0
      ? rawReasons.slice(0, 6).map((r, idx) => {
          const normalized = reasonLabel(r, idx);
          return {
            rank: idx + 1,
            label: normalized.label,
            description: normalized.description,
            factorType: factorTypes[idx % factorTypes.length],
          };
        })
      : [
          {
            rank: 1,
            label: 'Model-ranked opportunity',
            description: 'This match was selected by the hybrid ranking pipeline.',
            factorType: 'buyerActivity',
          },
        ];

  const companySize: MatchCardDTO['importerOrg']['companySize'] =
    trustScore >= 85 ? 'Enterprise' : trustScore >= 70 ? 'Mid-market' : 'SME';

  return {
    matchId: exporterId,
    status: 'active',
    importerOrg: {
      id: exporterId,
      name: exporterId || 'Unknown Exporter',
      country: card.exporter_state || 'Unknown',
      industry: card.exporter_cert || 'General',
      companySize,
    },
    importerLocation: {
      city: card.exporter_state || 'Unknown',
      region: card.exporter_state || 'Unknown',
      country: card.exporter_state || 'Unknown',
    },
    importerProducts: [
      {
        hsCode: 'N/A',
        category: card.exporter_cert || 'General',
        description: 'Mapped from ML pipeline exporter metadata.',
      },
    ],
    matchScore,
    trustScore,
    finalScore,
    factors,
    trustComponents,
    reasons,
    quantity: 'Based on model signals',
    timeline: 'TBD',
    lastActive: 'recent',
    buyerId,
    exporterId,
    backendWarning: card.warning || undefined,
  };
}

async function resolveBuyerId(explicitBuyerId: string): Promise<string> {
  const fromQuery = explicitBuyerId.trim();
  if (fromQuery) return fromQuery;

  const fromEnv = DEFAULT_BUYER_ID.trim();
  if (fromEnv) return fromEnv;

  const buyersRes = await fetch(`${ML_API_BASE_URL}/buyers?limit=1&offset=0`, {
    method: 'GET',
    cache: 'no-store',
  });

  if (!buyersRes.ok) return '';

  const buyersData = (await buyersRes.json()) as {
    items?: Array<{ Buyer_ID?: string }>;
  };
  const first = buyersData.items?.[0]?.Buyer_ID;
  return String(first || '').trim();
}

/**
 * GET /api/matches/feed
 *
 * Returns ML-ranked match cards mapped into frontend DTO shape.
 * Query params: buyerId, limit, offset, minMatchScore, minTrustScore
 *
 * Flow:
 *  - Resolve buyerId from query/env (or first buyer from backend)
 *  - Read /feed from FastAPI backend
 *  - Adapt backend card fields to MatchCardDTO used by UI components
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const limit = parseInt(searchParams.get('limit') || '10', 10);
  const offset = parseInt(searchParams.get('offset') || '0', 10);
  const minMatchScore = parseInt(searchParams.get('minMatchScore') || '0', 10);
  const minTrustScore = parseInt(searchParams.get('minTrustScore') || '0', 10);
  const requestedBuyerId = searchParams.get('buyerId') || '';

  try {
    const buyerId = await resolveBuyerId(requestedBuyerId);
    if (!buyerId) {
      return NextResponse.json(
        { error: 'No buyerId provided and no default buyer could be resolved.' },
        { status: 400 }
      );
    }

    const backendLimit = Math.max(limit + offset, limit, 1);
    const backendRes = await fetch(
      `${ML_API_BASE_URL}/feed?buyer_id=${encodeURIComponent(buyerId)}&limit=${backendLimit}`,
      {
        method: 'GET',
        cache: 'no-store',
      }
    );

    if (!backendRes.ok) {
      const details = await backendRes.text();
      return NextResponse.json(
        {
          error: 'Failed to fetch feed from ML backend.',
          details: details || backendRes.statusText,
        },
        { status: 502 }
      );
    }

    const backendPayload = (await backendRes.json()) as {
      cards?: BackendFeedCard[];
    };

    const mappedCards = (backendPayload.cards || []).map((card) => mapBackendCard(card, buyerId));

    const filtered = mappedCards.filter(
      (card) => card.matchScore >= minMatchScore && card.trustScore >= minTrustScore
    );

    const paged = filtered.slice(offset, offset + limit);

    const response: FeedResponse = {
      cards: paged,
      total: filtered.length,
      hasMore: offset + limit < filtered.length,
    };

    return NextResponse.json({ ...response, buyerId });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unexpected server error.';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
