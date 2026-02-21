
create table if not exists public.trade_products (
  id uuid primary key default gen_random_uuid(),
  org_id uuid not null references public.organizations(id) on delete cascade,
  hs_code text not null,
  category text not null,
  description text,
  created_at timestamptz not null default now()
);

alter table public.trade_products enable row level security;


create table if not exists public.trade_preferences (
  id uuid primary key default gen_random_uuid(),
  org_id uuid not null references public.organizations(id) on delete cascade,
  type text not null check (type in ('export_market', 'import_market', 'incoterm', 'purchase_frequency')),
  value text not null,
  created_at timestamptz not null default now()
);

alter table public.trade_preferences enable row level security;


create policy "trade_products_select" on public.trade_products
  for select using (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = trade_products.org_id
        and organization_members.user_id = auth.uid()
    )
  );

create policy "trade_products_insert" on public.trade_products
  for insert with check (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = trade_products.org_id
        and organization_members.user_id = auth.uid()
    )
  );

create policy "trade_products_update" on public.trade_products
  for update using (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = trade_products.org_id
        and organization_members.user_id = auth.uid()
    )
  );

create policy "trade_products_delete" on public.trade_products
  for delete using (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = trade_products.org_id
        and organization_members.user_id = auth.uid()
    )
  );


create policy "trade_prefs_select" on public.trade_preferences
  for select using (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = trade_preferences.org_id
        and organization_members.user_id = auth.uid()
    )
  );

create policy "trade_prefs_insert" on public.trade_preferences
  for insert with check (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = trade_preferences.org_id
        and organization_members.user_id = auth.uid()
    )
  );

create policy "trade_prefs_update" on public.trade_preferences
  for update using (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = trade_preferences.org_id
        and organization_members.user_id = auth.uid()
    )
  );

create policy "trade_prefs_delete" on public.trade_preferences
  for delete using (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = trade_preferences.org_id
        and organization_members.user_id = auth.uid()
    )
  );
