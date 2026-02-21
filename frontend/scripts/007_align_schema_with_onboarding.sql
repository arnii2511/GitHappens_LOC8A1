-- This migration aligns the database schema with what the onboarding code expects.
-- The onboarding uses simpler user-centric tables rather than the org-member model.

-- ============================================================
-- 1. Fix organizations table to match onboarding inserts
-- The onboarding inserts: user_id, company_name, registration_number,
-- country, city, address, website, year_established, number_of_employees,
-- annual_revenue, description
-- ============================================================

-- Add missing columns to organizations (keep existing ones)
ALTER TABLE public.organizations
  ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  ADD COLUMN IF NOT EXISTS company_name text,
  ADD COLUMN IF NOT EXISTS registration_number text,
  ADD COLUMN IF NOT EXISTS address text,
  ADD COLUMN IF NOT EXISTS year_established integer,
  ADD COLUMN IF NOT EXISTS number_of_employees text,
  ADD COLUMN IF NOT EXISTS annual_revenue text,
  ADD COLUMN IF NOT EXISTS description text;

-- Make name nullable since onboarding uses company_name instead
ALTER TABLE public.organizations ALTER COLUMN name DROP NOT NULL;
ALTER TABLE public.organizations ALTER COLUMN industry DROP NOT NULL;
ALTER TABLE public.organizations ALTER COLUMN company_size DROP DEFAULT;

-- Remove the company_size check constraint so NULL is allowed
ALTER TABLE public.organizations DROP CONSTRAINT IF EXISTS organizations_company_size_check;

-- Allow the onboarding user to insert orgs directly (with user_id)
DROP POLICY IF EXISTS "org_insert_authenticated" ON public.organizations;
CREATE POLICY "org_insert_authenticated" ON public.organizations
  FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Allow user to select their own org by user_id
DROP POLICY IF EXISTS "org_select_by_user" ON public.organizations;
CREATE POLICY "org_select_by_user" ON public.organizations
  FOR SELECT USING (auth.uid() = user_id);

-- Allow user to update their own org by user_id
DROP POLICY IF EXISTS "org_update_by_user" ON public.organizations;
CREATE POLICY "org_update_by_user" ON public.organizations
  FOR UPDATE USING (auth.uid() = user_id);

-- ============================================================
-- 2. Create trade_info table (the onboarding inserts into this)
-- ============================================================

CREATE TABLE IF NOT EXISTS public.trade_info (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  product_categories text[] DEFAULT '{}',
  primary_markets text[] DEFAULT '{}',
  certifications text[] DEFAULT '{}',
  min_order_value text,
  preferred_payment_terms text[] DEFAULT '{}',
  preferred_incoterms text[] DEFAULT '{}',
  logistics_capabilities text[] DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.trade_info ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "trade_info_select_own" ON public.trade_info;
CREATE POLICY "trade_info_select_own" ON public.trade_info
  FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "trade_info_insert_own" ON public.trade_info;
CREATE POLICY "trade_info_insert_own" ON public.trade_info
  FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "trade_info_update_own" ON public.trade_info;
CREATE POLICY "trade_info_update_own" ON public.trade_info
  FOR UPDATE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "trade_info_delete_own" ON public.trade_info;
CREATE POLICY "trade_info_delete_own" ON public.trade_info
  FOR DELETE USING (auth.uid() = user_id);

-- ============================================================
-- 3. Fix kyc_documents table to match onboarding inserts
-- The onboarding inserts: user_id, document_type, file_name,
-- file_path, file_size, mime_type
-- ============================================================

-- Add user_id and document_type columns
ALTER TABLE public.kyc_documents
  ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  ADD COLUMN IF NOT EXISTS document_type text;

-- Make org_id nullable (onboarding doesn't use it)
ALTER TABLE public.kyc_documents ALTER COLUMN org_id DROP NOT NULL;

-- Make uploaded_by nullable
ALTER TABLE public.kyc_documents ALTER COLUMN uploaded_by DROP NOT NULL;

-- Remove strict doc_type check so 'other' value works
ALTER TABLE public.kyc_documents DROP CONSTRAINT IF EXISTS kyc_documents_doc_type_check;

-- Make doc_type nullable since onboarding uses document_type instead
ALTER TABLE public.kyc_documents ALTER COLUMN doc_type DROP NOT NULL;

-- RLS: allow users to insert their own docs by user_id
DROP POLICY IF EXISTS "kyc_docs_insert_own" ON public.kyc_documents;
CREATE POLICY "kyc_docs_insert_own" ON public.kyc_documents
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- RLS: allow users to select their own docs by user_id
DROP POLICY IF EXISTS "kyc_docs_select_own" ON public.kyc_documents;
CREATE POLICY "kyc_docs_select_own" ON public.kyc_documents
  FOR SELECT USING (auth.uid() = user_id);

-- RLS: allow users to delete their own docs by user_id
DROP POLICY IF EXISTS "kyc_docs_delete_own" ON public.kyc_documents;
CREATE POLICY "kyc_docs_delete_own" ON public.kyc_documents
  FOR DELETE USING (auth.uid() = user_id);

-- ============================================================
-- 4. Fix storage bucket name and policies
-- The onboarding uploads to 'kyc-documents' not 'kyc-docs'
-- ============================================================

INSERT INTO storage.buckets (id, name, public)
VALUES ('kyc-documents', 'kyc-documents', false)
ON CONFLICT (id) DO NOTHING;

-- Storage policy: authenticated users can upload to their own user folder
DROP POLICY IF EXISTS "kyc_documents_storage_insert" ON storage.objects;
CREATE POLICY "kyc_documents_storage_insert" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'kyc-documents'
    AND auth.role() = 'authenticated'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );

-- Storage policy: authenticated users can read their own files
DROP POLICY IF EXISTS "kyc_documents_storage_select" ON storage.objects;
CREATE POLICY "kyc_documents_storage_select" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'kyc-documents'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );

-- Storage policy: authenticated users can delete their own files
DROP POLICY IF EXISTS "kyc_documents_storage_delete" ON storage.objects;
CREATE POLICY "kyc_documents_storage_delete" ON storage.objects
  FOR DELETE USING (
    bucket_id = 'kyc-documents'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );
