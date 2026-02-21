// ─── Core Domain Types (matching DB schema from blueprint) ───

export type MatchStatus = 'active' | 'blocked' | 'expired' | 'converted';
export type SwipeAction = 'like' | 'dislike' | 'save' | 'block';
export type OutreachChannel = 'email' | 'linkedin' | 'whatsapp';

// Organization (importer / exporter)
export interface Organization {
  id: string;
  name: string;
  country: string;
  industry: string;
  companySize: 'SME' | 'Mid-market' | 'Enterprise';
  logoUrl?: string;
  website?: string;
}

// Product alignment info
export interface ProductInfo {
  hsCode: string;
  category: string;
  description: string;
}

// Location info (optional geo UX)
export interface CompanyLocation {
  city: string;
  region: string;
  country: string;
}

// Score factor contributions (from match_score_snapshots)
export interface ScoreFactors {
  productFit: number;    // 0-100
  geographyFit: number;  // 0-100
  buyerActivity: number; // 0-100
  scaleFit: number;      // 0-100
  marketTrend: number;   // 0-100
  tradeFrequency: number;// 0-100
}

// Trust score components (RV/TH/DF/PB)
export interface TrustComponents {
  registrationVerification: number;  // RV 0-100
  tradeHistory: number;              // TH 0-100
  documentationFidelity: number;     // DF 0-100
  paymentBehavior: number;           // PB 0-100
}

// Match reason (from match_reasons)
export interface MatchReason {
  rank: number;
  label: string;
  description: string;
  factorType: keyof ScoreFactors;
}

// Snapshot for time-series score tracking
export interface ScoreSnapshot {
  computedAt: string;
  matchScore: number;
  trustScore: number;
  finalScore: number;
  factors: ScoreFactors;
}

// The main Match Card DTO (what the feed API returns)
export interface MatchCardDTO {
  matchId: string;
  status: MatchStatus;

  // Importer org details
  importerOrg: Organization;
  importerLocation?: CompanyLocation;
  importerProducts: ProductInfo[];

  // Scores (denormalized from matches table)
  matchScore: number;
  trustScore: number;
  finalScore: number;

  // Breakdown (from latest match_score_snapshots)
  factors: ScoreFactors;
  trustComponents: TrustComponents;

  // Explainability (from match_reasons)
  reasons: MatchReason[];

  // Buyer interest signals
  quantity: string;
  timeline: string;
  lastActive: string;
}

// Swipe record
export interface Swipe {
  id: string;
  userId: string;
  matchId: string;
  action: SwipeAction;
  createdAt: string;
}

// Outreach draft (generated on Connect)
export interface OutreachDraft {
  id: string;
  matchId: string;
  channel: OutreachChannel;
  subject: string;
  message: string;
  personalization: Record<string, string>;
  createdAt: string;
}

// ─── Connection & Chat Types ───

export type ConnectionStatus = 'pending' | 'accepted' | 'declined';

export interface ConnectionRequest {
  id: string;
  matchId: string;
  exporterOrgId: string;
  exporterOrgName: string;
  importerOrgId: string;
  importerOrgName: string;
  importerCountry: string;
  importerIndustry: string;
  finalScore: number;
  status: ConnectionStatus;
  createdAt: string;
  respondedAt?: string;
  note?: string;
}

export interface ChatMessage {
  id: string;
  connectionId: string;
  senderId: string;
  senderName: string;
  senderRole: 'exporter' | 'importer';
  content: string;
  createdAt: string;
  read: boolean;
}

export type NotificationType =
  | 'connection_sent'
  | 'connection_accepted'
  | 'connection_declined'
  | 'new_message';

export interface AppNotification {
  id: string;
  type: NotificationType;
  title: string;
  description: string;
  connectionId: string;
  read: boolean;
  createdAt: string;
}

// Feed request params
export interface FeedParams {
  exporterOrgId: string;
  limit: number;
  offset: number;
  minMatchScore?: number;
  minTrustScore?: number;
  countryPreferences?: string[];
}

// Feed response
export interface FeedResponse {
  cards: MatchCardDTO[];
  total: number;
  hasMore: boolean;
}

// Swipe response
export interface SwipeResponse {
  success: boolean;
  action: SwipeAction;
  matchId: string;
  outreachDraftId?: string;
}
