import { NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';

type ConnectionRow = {
  id: string;
  match_id: string | null;
  exporter_org_id: string | null;
  exporter_org_name: string | null;
  importer_org_id: string | null;
  importer_org_name: string | null;
  importer_country: string | null;
  importer_industry: string | null;
  final_score: number | null;
  status: 'pending' | 'accepted' | 'declined';
  created_at: string;
  responded_at: string | null;
  note: string | null;
};

function toConnectionDTO(row: ConnectionRow) {
  return {
    id: row.id,
    matchId: row.match_id ?? '',
    exporterOrgId: row.exporter_org_id ?? '',
    exporterOrgName: row.exporter_org_name ?? 'Unknown Exporter',
    importerOrgId: row.importer_org_id ?? '',
    importerOrgName: row.importer_org_name ?? 'Unknown Importer',
    importerCountry: row.importer_country ?? '',
    importerIndustry: row.importer_industry ?? '',
    finalScore: Number(row.final_score ?? 0),
    status: row.status,
    createdAt: row.created_at,
    respondedAt: row.responded_at ?? undefined,
    note: row.note ?? undefined,
  };
}

// GET /api/connections - List connections for the logged-in user
export async function GET() {
  const supabase = await createClient();
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();

  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { data: profile } = await supabase
    .from('profiles')
    .select('account_type')
    .eq('id', user.id)
    .maybeSingle();

  let rows: ConnectionRow[] = [];
  if (profile?.account_type === 'buyer') {
    const { data, error } = await supabase
      .from('connections')
      .select('*')
      .eq('importer_user_id', user.id)
      .order('created_at', { ascending: false });
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    rows = (data as ConnectionRow[]) ?? [];

    const { data: mappedRows } = await supabase
      .from('buyer_directory')
      .select('ml_buyer_id')
      .eq('user_id', user.id);
    const mappedIds = (mappedRows ?? []).map((r) => String((r as { ml_buyer_id?: string }).ml_buyer_id || '').trim()).filter(Boolean);
    if (mappedIds.length > 0) {
      const { data: unresolved } = await supabase
        .from('connections')
        .select('*')
        .is('importer_user_id', null)
        .in('importer_org_id', mappedIds)
        .order('created_at', { ascending: false });
      const map = new Map<string, ConnectionRow>();
      for (const r of rows) map.set(r.id, r);
      for (const r of ((unresolved as ConnectionRow[]) ?? [])) map.set(r.id, r);
      rows = Array.from(map.values()).sort((a, b) => +new Date(b.created_at) - +new Date(a.created_at));
    }
  } else {
    const { data, error } = await supabase
      .from('connections')
      .select('*')
      .eq('exporter_user_id', user.id)
      .order('created_at', { ascending: false });
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    rows = (data as ConnectionRow[]) ?? [];
  }

  return NextResponse.json({
    connections: rows.map(toConnectionDTO),
    total: rows.length,
  });
}

// POST /api/connections - Create a connection request
export async function POST(request: Request) {
  const supabase = await createClient();
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();

  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = (await request.json()) as {
    matchId?: string;
    importerOrgId?: string;
    importerOrgName?: string;
    importerCountry?: string;
    importerIndustry?: string;
    finalScore?: number;
    note?: string;
  };

  const { data: exporterOrg } = await supabase
    .from('organizations')
    .select('id, company_name')
    .eq('user_id', user.id)
    .maybeSingle();

  let importerUserId: string | null = null;
  if (body.importerOrgId) {
    const { data: byOrgId } = await supabase
      .from('organizations')
      .select('user_id')
      .eq('id', body.importerOrgId)
      .maybeSingle();
    importerUserId = byOrgId?.user_id ?? null;
  }
  if (!importerUserId && body.importerOrgName) {
    const { data: byName } = await supabase
      .from('organizations')
      .select('user_id')
      .eq('company_name', body.importerOrgName)
      .maybeSingle();
    importerUserId = byName?.user_id ?? null;
  }
  if (!importerUserId && (body.importerOrgId || body.importerOrgName)) {
    const mlId = String(body.importerOrgId || body.importerOrgName || '').trim();
    if (mlId) {
      const { data: mapped } = await supabase
        .from('buyer_directory')
        .select('user_id')
        .eq('ml_buyer_id', mlId)
        .maybeSingle();
      importerUserId = mapped?.user_id ?? null;
    }
  }

  const insertPayload = {
    match_id: body.matchId ?? null,
    exporter_user_id: user.id,
    exporter_org_id: exporterOrg?.id ?? null,
    exporter_org_name: exporterOrg?.company_name ?? 'Your Company',
    importer_user_id: importerUserId,
    importer_org_id: body.importerOrgId ?? null,
    importer_org_name: body.importerOrgName ?? 'Unknown Importer',
    importer_country: body.importerCountry ?? '',
    importer_industry: body.importerIndustry ?? '',
    final_score: Number(body.finalScore ?? 0),
    status: 'pending' as const,
    note: body.note ?? null,
  };

  const { data, error } = await supabase
    .from('connections')
    .insert(insertPayload)
    .select('*')
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  const notificationsToInsert: Array<{
    user_id: string;
    type: 'connection_sent';
    title: string;
    description: string;
    connection_id: string;
    read: boolean;
  }> = [
    {
      user_id: user.id,
      type: 'connection_sent',
      title: 'Connection Sent',
      description: `Your connection request to ${insertPayload.importer_org_name} was sent.`,
      connection_id: data.id,
      read: true,
    },
  ];

  if (importerUserId) {
    notificationsToInsert.push({
      user_id: importerUserId,
      type: 'connection_sent',
      title: 'New Connection Request',
      description: `${insertPayload.exporter_org_name} sent you a connection request.`,
      connection_id: data.id,
      read: false,
    });
  }

  await supabase.from('app_notifications').insert(notificationsToInsert);

  return NextResponse.json({
    success: true,
    connection: toConnectionDTO(data as ConnectionRow),
  });
}
