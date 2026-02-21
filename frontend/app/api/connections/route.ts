import { NextResponse } from 'next/server';
import { mockConnections } from '@/lib/mock-data';

// GET /api/connections - List all connections
export async function GET() {
  return NextResponse.json({
    connections: mockConnections,
    total: mockConnections.length,
  });
}

// POST /api/connections - Create a new connection request
export async function POST(request: Request) {
  const body = await request.json();
  const { matchId, importerOrgId, importerOrgName, note } = body;

  const newConnection = {
    id: `conn-${Date.now()}`,
    matchId,
    exporterOrgId: 'org-exp-001',
    exporterOrgName: 'Your Company',
    importerOrgId,
    importerOrgName,
    importerCountry: '',
    importerIndustry: '',
    finalScore: 0,
    status: 'pending' as const,
    createdAt: new Date().toISOString(),
    note,
  };

  return NextResponse.json({
    success: true,
    connection: newConnection,
  });
}
