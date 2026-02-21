import { NextRequest, NextResponse } from 'next/server';
import { mockMatchCards } from '@/lib/mock-data';
import type { FeedResponse } from '@/lib/types';

/**
 * GET /api/matches/feed
 *
 * Returns the match feed for the current exporter org.
 * Query params: exporterOrgId, limit, offset, minMatchScore, minTrustScore
 *
 * In production this would:
 *  - Verify user session via organization_members
 *  - Read user_preferences for filters/weights
 *  - Query matches + organizations + match_score_snapshots + match_reasons
 *  - Exclude already-swiped matches via swipes table
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const limit = parseInt(searchParams.get('limit') || '10', 10);
  const offset = parseInt(searchParams.get('offset') || '0', 10);
  const minMatchScore = parseInt(searchParams.get('minMatchScore') || '0', 10);
  const minTrustScore = parseInt(searchParams.get('minTrustScore') || '0', 10);

  // Filter mock data (simulating DB filters)
  let filtered = mockMatchCards.filter(
    (card) =>
      card.status === 'active' &&
      card.matchScore >= minMatchScore &&
      card.trustScore >= minTrustScore
  );

  // Sort by finalScore descending (highest first)
  filtered.sort((a, b) => b.finalScore - a.finalScore);

  // Paginate
  const total = filtered.length;
  const paged = filtered.slice(offset, offset + limit);

  const response: FeedResponse = {
    cards: paged,
    total,
    hasMore: offset + limit < total,
  };

  return NextResponse.json(response);
}
