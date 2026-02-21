import { createClient } from '@/lib/supabase/server'
import { NextResponse } from 'next/server'

export async function GET() {
  const supabase = await createClient()

  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser()

  console.log('[v0] Profile GET - user:', user?.id, 'authErr:', authErr?.message)

  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  // Fetch profile (maybeSingle so it returns null instead of error if missing)
  const { data: profile, error: profileErr } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', user.id)
    .maybeSingle()

  if (profileErr) {
    return NextResponse.json({ error: profileErr.message }, { status: 500 })
  }

  // Fetch organization
  const { data: org } = await supabase
    .from('organizations')
    .select('*')
    .eq('user_id', user.id)
    .maybeSingle()

  // Fetch trade info
  const { data: tradeInfo } = await supabase
    .from('trade_info')
    .select('*')
    .eq('user_id', user.id)
    .maybeSingle()

  // Fetch KYC documents
  const { data: documents } = await supabase
    .from('kyc_documents')
    .select('*')
    .eq('user_id', user.id)
    .order('created_at', { ascending: false })

  return NextResponse.json({
    email: user.email,
    profile,
    organization: org,
    tradeInfo,
    documents: documents ?? [],
  })
}

export async function PUT(request: Request) {
  const supabase = await createClient()

  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser()

  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const body = await request.json()
  const { section, data } = body as { section: string; data: Record<string, unknown> }

  try {
    switch (section) {
      case 'profile': {
        const { error } = await supabase
          .from('profiles')
          .update({
            full_name: data.full_name,
            phone: data.phone,
            account_type: data.account_type,
            updated_at: new Date().toISOString(),
          })
          .eq('id', user.id)
        if (error) throw error
        break
      }

      case 'organization': {
        const { error } = await supabase
          .from('organizations')
          .update({
            company_name: data.company_name,
            registration_number: data.registration_number || null,
            country: data.country,
            city: data.city || null,
            address: data.address || null,
            website: data.website || null,
            year_established: data.year_established ? String(data.year_established) : null,
            number_of_employees: data.number_of_employees || null,
            annual_revenue: data.annual_revenue || null,
            description: data.description || null,
            updated_at: new Date().toISOString(),
          })
          .eq('user_id', user.id)
        if (error) throw error
        break
      }

      case 'trade': {
        const { error } = await supabase
          .from('trade_info')
          .update({
            product_categories: data.product_categories,
            primary_markets: data.primary_markets,
            certifications: data.certifications,
            min_order_value: data.min_order_value || null,
            preferred_payment_terms: data.preferred_payment_terms,
            preferred_incoterms: data.preferred_incoterms,
            logistics_capabilities: data.logistics_capabilities,
            updated_at: new Date().toISOString(),
          })
          .eq('user_id', user.id)
        if (error) throw error
        break
      }

      default:
        return NextResponse.json({ error: 'Invalid section' }, { status: 400 })
    }

    return NextResponse.json({ success: true })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Update failed'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
