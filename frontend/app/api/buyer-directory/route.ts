import { NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';

export async function GET() {
  const supabase = await createClient();
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();
  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { data, error } = await supabase
    .from('buyer_directory')
    .select('id, ml_buyer_id, org_id, org_name, created_at')
    .eq('user_id', user.id)
    .order('created_at', { ascending: false });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ items: data ?? [] });
}

export async function POST(request: Request) {
  const supabase = await createClient();
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();
  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = (await request.json()) as { mlBuyerId?: string };
  const mlBuyerId = String(body.mlBuyerId || '').trim();
  if (!mlBuyerId) {
    return NextResponse.json({ error: 'mlBuyerId is required' }, { status: 400 });
  }

  const { data: org } = await supabase
    .from('organizations')
    .select('id, company_name')
    .eq('user_id', user.id)
    .maybeSingle();

  const { data, error } = await supabase
    .from('buyer_directory')
    .upsert(
      {
        ml_buyer_id: mlBuyerId,
        user_id: user.id,
        org_id: org?.id ?? null,
        org_name: org?.company_name ?? null,
        updated_at: new Date().toISOString(),
      },
      { onConflict: 'ml_buyer_id' }
    )
    .select('*')
    .single();
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ success: true, item: data });
}
