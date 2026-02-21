-- Create organizations table
create table if not exists public.organizations (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  country text not null,
  city text,
  website text,
  industry text not null,
  company_size text not null check (company_size in ('1-10', '11-50', '51-200', '201-500', '500+')),
  annual_trade_volume text,
  created_at timestamptz not null default now()
);

alter table public.organizations enable row level security;

-- Create organization_members join table
create table if not exists public.organization_members (
  id uuid primary key default gen_random_uuid(),
  org_id uuid not null references public.organizations(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null default 'owner',
  unique(org_id, user_id)
);

alter table public.organization_members enable row level security;

-- RLS for organizations: only members can view/update
create policy "org_select_member" on public.organizations
  for select using (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = organizations.id
        and organization_members.user_id = auth.uid()
    )
  );

create policy "org_insert_authenticated" on public.organizations
  for insert with check (auth.role() = 'authenticated');

create policy "org_update_member" on public.organizations
  for update using (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = organizations.id
        and organization_members.user_id = auth.uid()
    )
  );

-- RLS for organization_members
create policy "org_members_select_own" on public.organization_members
  for select using (auth.uid() = user_id);

create policy "org_members_insert_own" on public.organization_members
  for insert with check (auth.uid() = user_id);

create policy "org_members_delete_own" on public.organization_members
  for delete using (auth.uid() = user_id);
