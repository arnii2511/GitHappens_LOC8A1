
ALTER TABLE public.organizations
  ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  ADD COLUMN IF NOT EXISTS company_name text,
  ADD COLUMN IF NOT EXISTS registration_number text,
  ADD COLUMN IF NOT EXISTS address text,
  ADD COLUMN IF NOT EXISTS year_established integer,
  ADD COLUMN IF NOT EXISTS number_of_employees text,
  ADD COLUMN IF NOT EXISTS annual_revenue text,
  ADD COLUMN IF NOT EXISTS description text;


ALTER TABLE public.organizations ALTER COLUMN name DROP NOT NULL;
ALTER TABLE public.organizations ALTER COLUMN industry DROP NOT NULL;
ALTER TABLE public.organizations ALTER COLUMN company_size DROP DEFAULT;


ALTER TABLE public.organizations DROP CONSTRAINT IF EXISTS organizations_company_size_check;


DROP POLICY IF EXISTS "org_insert_authenticated" ON public.organizations;
CREATE POLICY "org_insert_authenticated" ON public.organizations
  FOR INSERT WITH CHECK (auth.role() = 'authenticated');


DROP POLICY IF EXISTS "org_select_by_user" ON public.organizations;
CREATE POLICY "org_select_by_user" ON public.organizations
  FOR SELECT USING (auth.uid() = user_id);


DROP POLICY IF EXISTS "org_update_by_user" ON public.organizations;
CREATE POLICY "org_update_by_user" ON public.organizations
  FOR UPDATE USING (auth.uid() = user_id);



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


ALTER TABLE public.kyc_documents
  ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  ADD COLUMN IF NOT EXISTS document_type text;


ALTER TABLE public.kyc_documents ALTER COLUMN org_id DROP NOT NULL;


ALTER TABLE public.kyc_documents ALTER COLUMN uploaded_by DROP NOT NULL;


ALTER TABLE public.kyc_documents DROP CONSTRAINT IF EXISTS kyc_documents_doc_type_check;


ALTER TABLE public.kyc_documents ALTER COLUMN doc_type DROP NOT NULL;


DROP POLICY IF EXISTS "kyc_docs_insert_own" ON public.kyc_documents;
CREATE POLICY "kyc_docs_insert_own" ON public.kyc_documents
  FOR INSERT WITH CHECK (auth.uid() = user_id);


DROP POLICY IF EXISTS "kyc_docs_select_own" ON public.kyc_documents;
CREATE POLICY "kyc_docs_select_own" ON public.kyc_documents
  FOR SELECT USING (auth.uid() = user_id);


DROP POLICY IF EXISTS "kyc_docs_delete_own" ON public.kyc_documents;
CREATE POLICY "kyc_docs_delete_own" ON public.kyc_documents
  FOR DELETE USING (auth.uid() = user_id);


INSERT INTO storage.buckets (id, name, public)
VALUES ('kyc-documents', 'kyc-documents', false)
ON CONFLICT (id) DO NOTHING;


DROP POLICY IF EXISTS "kyc_documents_storage_insert" ON storage.objects;
CREATE POLICY "kyc_documents_storage_insert" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'kyc-documents'
    AND auth.role() = 'authenticated'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );


DROP POLICY IF EXISTS "kyc_documents_storage_select" ON storage.objects;
CREATE POLICY "kyc_documents_storage_select" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'kyc-documents'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );


DROP POLICY IF EXISTS "kyc_documents_storage_delete" ON storage.objects;
CREATE POLICY "kyc_documents_storage_delete" ON storage.objects
  FOR DELETE USING (
    bucket_id = 'kyc-documents'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );
