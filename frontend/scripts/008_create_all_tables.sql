-- ============================================================
-- Full schema creation: profiles, organizations, trade_info,
-- kyc_documents + storage bucket + RLS + auto-profile trigger
-- ============================================================

-- 1. PROFILES TABLE
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  account_type text not null default 'exporter' check (account_type in ('exporter', 'buyer')),
  full_name text,
  phone text,
  onboarding_completed boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

create policy "Users can read own profile"
  on public.profiles for select
  using (auth.uid() = id);

create policy "Users can update own profile"
  on public.profiles for update
  using (auth.uid() = id);

create policy "Users can insert own profile"
  on public.profiles for insert
  with check (auth.uid() = id);


-- 2. ORGANIZATIONS TABLE
create table if not exists public.organizations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  company_name text not null,
  registration_number text,
  country text not null,
  city text,
  address text,
  website text,
  year_established integer,
  number_of_employees text,
  annual_revenue text,
  description text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.organizations enable row level security;

create policy "Users can read own organizations"
  on public.organizations for select
  using (auth.uid() = user_id);

create policy "Users can insert own organizations"
  on public.organizations for insert
  with check (auth.uid() = user_id);

create policy "Users can update own organizations"
  on public.organizations for update
  using (auth.uid() = user_id);


-- 3. TRADE_INFO TABLE
create table if not exists public.trade_info (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  product_categories text[] not null default '{}',
  primary_markets text[] not null default '{}',
  certifications text[] not null default '{}',
  min_order_value text,
  preferred_payment_terms text[] not null default '{}',
  preferred_incoterms text[] not null default '{}',
  logistics_capabilities text[] not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.trade_info enable row level security;

create policy "Users can read own trade info"
  on public.trade_info for select
  using (auth.uid() = user_id);

create policy "Users can insert own trade info"
  on public.trade_info for insert
  with check (auth.uid() = user_id);

create policy "Users can update own trade info"
  on public.trade_info for update
  using (auth.uid() = user_id);


-- 4. KYC_DOCUMENTS TABLE
create table if not exists public.kyc_documents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  document_type text not null default 'other',
  file_name text not null,
  file_path text not null,
  file_size bigint,
  mime_type text,
  created_at timestamptz not null default now()
);

alter table public.kyc_documents enable row level security;

create policy "Users can read own kyc docs"
  on public.kyc_documents for select
  using (auth.uid() = user_id);

create policy "Users can insert own kyc docs"
  on public.kyc_documents for insert
  with check (auth.uid() = user_id);

create policy "Users can update own kyc docs"
  on public.kyc_documents for update
  using (auth.uid() = user_id);


-- 5. STORAGE BUCKET for KYC documents
insert into storage.buckets (id, name, public)
values ('kyc-documents', 'kyc-documents', false)
on conflict (id) do nothing;

-- Storage policies: users can upload to their own folder
create policy "Users can upload own KYC files"
  on storage.objects for insert
  with check (
    bucket_id = 'kyc-documents'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

create policy "Users can read own KYC files"
  on storage.objects for select
  using (
    bucket_id = 'kyc-documents'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

create policy "Users can delete own KYC files"
  on storage.objects for delete
  using (
    bucket_id = 'kyc-documents'
    and (storage.foldername(name))[1] = auth.uid()::text
  );


-- 6. AUTO-CREATE PROFILE on user signup
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, account_type, full_name, phone)
  values (
    new.id,
    coalesce(new.raw_user_meta_data ->> 'account_type', 'exporter'),
    coalesce(new.raw_user_meta_data ->> 'full_name', null),
    coalesce(new.raw_user_meta_data ->> 'phone', null)
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;

create trigger on_auth_user_created
  after insert on auth.users
  for each row
  execute function public.handle_new_user();
