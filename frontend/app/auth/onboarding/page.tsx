'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import {
  Globe,
  Loader2,
  User,
  Building2,
  Package,
  ShieldCheck,
  ArrowLeft,
  ArrowRight,
  Check,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import StepIndicator from '@/components/StepIndicator'
import FileUploadZone, { type UploadedFile } from '@/components/FileUploadZone'

/* ─── Constants ─── */

const STEPS = [
  { label: 'Profile' },
  { label: 'Organization' },
  { label: 'Trade Info' },
  { label: 'Documents' },
]

const COUNTRIES = [
  'United States', 'United Kingdom', 'China', 'India', 'Germany', 'Japan',
  'Brazil', 'South Korea', 'France', 'Canada', 'Australia', 'Mexico',
  'Indonesia', 'Turkey', 'Saudi Arabia', 'Nigeria', 'South Africa',
  'United Arab Emirates', 'Vietnam', 'Thailand', 'Malaysia', 'Philippines',
  'Egypt', 'Kenya', 'Colombia', 'Chile', 'Bangladesh', 'Pakistan', 'Other',
]

const PRODUCT_CATEGORIES = [
  'Agriculture & Food', 'Textiles & Apparel', 'Electronics & Electrical',
  'Machinery & Equipment', 'Chemicals & Petrochemicals', 'Construction Materials',
  'Metals & Mining', 'Automotive & Transport', 'Consumer Goods',
  'Pharmaceuticals & Medical', 'Energy & Renewables', 'Other',
]

const CERTIFICATIONS = [
  'ISO 9001', 'ISO 14001', 'ISO 22000', 'HACCP', 'CE Marking', 'FDA Approved',
  'Fair Trade', 'Organic Certified', 'GMP', 'BRC', 'BSCI', 'SA8000', 'Other',
]

const PAYMENT_TERMS = [
  'Letter of Credit (L/C)', 'Telegraphic Transfer (T/T)', 'Documentary Collection (D/P)',
  'Open Account', 'Cash in Advance', 'Consignment',
]

const INCOTERMS = [
  'EXW', 'FCA', 'FAS', 'FOB', 'CFR', 'CIF', 'CPT', 'CIP', 'DAP', 'DPU', 'DDP',
]

const LOGISTICS = [
  'Sea Freight (FCL)', 'Sea Freight (LCL)', 'Air Freight', 'Road Transport',
  'Rail Freight', 'Multimodal', 'Warehousing', 'Cold Chain',
]

/* ─── Types ─── */

interface ProfileForm {
  accountType: 'exporter' | 'buyer'
  fullName: string
  phone: string
}

interface OrgForm {
  companyName: string
  registrationNumber: string
  country: string
  city: string
  address: string
  website: string
  yearEstablished: string
  numberOfEmployees: string
  annualRevenue: string
  description: string
}

interface TradeForm {
  productCategories: string[]
  primaryMarkets: string[]
  certifications: string[]
  minOrderValue: string
  preferredPaymentTerms: string[]
  preferredIncoterms: string[]
  logisticsCapabilities: string[]
}

/* ─── Component ─── */

export default function OnboardingPage() {
  const router = useRouter()
  const [step, setStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [initializing, setInitializing] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [userId, setUserId] = useState<string | null>(null)

  // Step 1: Profile
  const [profile, setProfile] = useState<ProfileForm>({
    accountType: 'exporter',
    fullName: '',
    phone: '',
  })

  // Step 2: Organization
  const [org, setOrg] = useState<OrgForm>({
    companyName: '',
    registrationNumber: '',
    country: '',
    city: '',
    address: '',
    website: '',
    yearEstablished: '',
    numberOfEmployees: '',
    annualRevenue: '',
    description: '',
  })

  // Step 3: Trade Info
  const [trade, setTrade] = useState<TradeForm>({
    productCategories: [],
    primaryMarkets: [],
    certifications: [],
    minOrderValue: '',
    preferredPaymentTerms: [],
    preferredIncoterms: [],
    logisticsCapabilities: [],
  })

  // Step 4: KYC Documents
  const [files, setFiles] = useState<UploadedFile[]>([])

  // Check auth on mount
  useEffect(() => {
    async function init() {
      const supabase = createClient()
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        router.push('/auth/login')
        return
      }
      setUserId(user.id)

      // Check if already completed onboarding
      const { data: existingProfile } = await supabase
        .from('profiles')
        .select('onboarding_completed')
        .eq('id', user.id)
        .single()

      if (existingProfile?.onboarding_completed) {
        router.replace('/Dashboard')
        return
      }

      setInitializing(false)
    }
    init()
  }, [router])

  /* ─── Step Validation ─── */

  function isStepValid(s: number): boolean {
    switch (s) {
      case 0:
        return profile.fullName.trim() !== '' && profile.phone.trim() !== ''
      case 1:
        return org.companyName.trim() !== '' && org.country !== ''
      case 2:
        return trade.productCategories.length > 0 && trade.primaryMarkets.length > 0
      case 3:
        return true // Documents are optional
      default:
        return false
    }
  }

  /* ─── Multi-select toggle helper ─── */

  function toggleArrayItem(arr: string[], item: string): string[] {
    return arr.includes(item) ? arr.filter((x) => x !== item) : [...arr, item]
  }

  /* ─── Submit all steps ─── */

  async function handleFinish() {
    if (!userId) return
    setLoading(true)
    setError(null)

    try {
      const supabase = createClient()

      // Step 1: Update profile
      const { error: profileErr } = await supabase
        .from('profiles')
        .update({
          account_type: profile.accountType,
          full_name: profile.fullName,
          phone: profile.phone,
          updated_at: new Date().toISOString(),
        })
        .eq('id', userId)

      if (profileErr) throw profileErr

      // Step 2: Create or update organization
      const { data: orgData, error: orgErr } = await supabase
        .from('organizations')
        .upsert({
          user_id: userId,
          company_name: org.companyName,
          registration_number: org.registrationNumber || null,
          country: org.country,
          city: org.city || null,
          address: org.address || null,
          website: org.website || null,
          year_established: org.yearEstablished ? parseInt(org.yearEstablished) : null,
          number_of_employees: org.numberOfEmployees || null,
          annual_revenue: org.annualRevenue || null,
          description: org.description || null,
        }, { onConflict: 'user_id' })
        .select('id')
        .single()

      if (orgErr) throw orgErr

      // Step 3: Create or update trade info
      const { error: tradeErr } = await supabase.from('trade_info').upsert({
        user_id: userId,
        product_categories: trade.productCategories,
        primary_markets: trade.primaryMarkets,
        certifications: trade.certifications,
        min_order_value: trade.minOrderValue || null,
        preferred_payment_terms: trade.preferredPaymentTerms,
        preferred_incoterms: trade.preferredIncoterms,
        logistics_capabilities: trade.logisticsCapabilities,
      }, { onConflict: 'user_id' })

      if (tradeErr) throw tradeErr

      // Step 4: Upload KYC documents
      const validFiles = files.filter((f) => f.status !== 'error')
      for (const f of validFiles) {
        const filePath = `${userId}/${Date.now()}-${f.name}`
        const { error: uploadErr } = await supabase.storage
          .from('kyc-documents')
          .upload(filePath, f.file)

        if (uploadErr) {
          console.error('Upload error:', uploadErr)
          continue // Don't block onboarding if a file upload fails
        }

        await supabase.from('kyc_documents').insert({
          user_id: userId,
          document_type: 'other',
          file_name: f.name,
          file_path: filePath,
          file_size: f.size,
          mime_type: f.type,
        })
      }

      // Mark onboarding as complete
      const { error: completeErr } = await supabase
        .from('profiles')
        .update({
  onboarding_completed: true,
  onboarding_completed_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
})

        .eq('id', userId)

      if (completeErr) throw completeErr

      // Use orgData to avoid unused variable
      if (orgData?.id) {
        console.log('Organization created:', orgData.id)
      }

router.replace('/Dashboard')
      router.refresh()
    } catch (err) {
      console.error('Onboarding error:', err)
      setError(
        err instanceof Error ? err.message : 'An error occurred while saving. Please try again.',
      )
      setLoading(false)
    }
  }

  /* ─── Navigation ─── */

  function handleNext() {
    if (step < STEPS.length - 1) {
      setStep(step + 1)
    } else {
      handleFinish()
    }
  }

  function handleBack() {
    if (step > 0) setStep(step - 1)
  }

  /* ─── Render ─── */

  if (initializing) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-cyan-500 rounded-lg blur opacity-50" />
            <div className="relative bg-card px-3 py-2 rounded-lg flex items-center gap-2">
              <Globe className="w-5 h-5 text-secondary" />
              <span className="font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                NexPort
              </span>
            </div>
          </div>
          <span className="text-muted-foreground text-sm">Account Setup</span>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 container mx-auto px-4 py-8 max-w-2xl">
        {/* Step Indicator */}
        <div className="mb-8">
          <StepIndicator steps={STEPS} currentStep={step} />
        </div>

        {error && (
          <div className="mb-6 bg-destructive/10 border border-destructive/30 text-destructive-foreground text-sm rounded-lg p-3">
            {error}
          </div>
        )}

        {/* Step content */}
        <div className="bg-card border border-border rounded-xl p-6 space-y-6">
          {step === 0 && (
            <StepProfile profile={profile} onChange={setProfile} />
          )}
          {step === 1 && <StepOrganization org={org} onChange={setOrg} />}
          {step === 2 && (
            <StepTradeInfo
              trade={trade}
              onChange={setTrade}
              toggleArrayItem={toggleArrayItem}
            />
          )}
          {step === 3 && <StepDocuments files={files} onFilesChange={setFiles} />}
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between mt-6">
          <Button
            type="button"
            variant="outline"
            onClick={handleBack}
            disabled={step === 0}
            className="border-border text-foreground"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>

          <Button
            type="button"
            onClick={handleNext}
            disabled={!isStepValid(step) || loading}
            className="bg-primary hover:bg-primary/90 text-primary-foreground"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : step === STEPS.length - 1 ? (
              <>
                <Check className="w-4 h-4 mr-2" />
                Complete Setup
              </>
            ) : (
              <>
                Next
                <ArrowRight className="w-4 h-4 ml-2" />
              </>
            )}
          </Button>
        </div>
      </main>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════
   Step Sub-Components
   ═══════════════════════════════════════════════════════════════════ */

/* ─── Step 1: Profile ─── */

function StepProfile({
  profile,
  onChange,
}: {
  profile: ProfileForm
  onChange: (p: ProfileForm) => void
}) {
  return (
    <>
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
          <User className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-foreground">Your Profile</h2>
          <p className="text-sm text-muted-foreground">
            Tell us about yourself and your role in trade
          </p>
        </div>
      </div>

      {/* Account Type */}
      <div className="space-y-2">
        <Label className="text-foreground">I am a...</Label>
        <div className="grid grid-cols-2 gap-3">
          {(['exporter', 'buyer'] as const).map((type) => (
            <button
              key={type}
              type="button"
              onClick={() => onChange({ ...profile, accountType: type })}
              className={`p-4 rounded-lg border text-left transition-colors ${
                profile.accountType === type
                  ? 'border-primary bg-primary/10 text-foreground'
                  : 'border-border bg-background text-muted-foreground hover:border-muted-foreground'
              }`}
            >
              <p className="font-medium capitalize text-sm">{type}</p>
              <p className="text-xs mt-1 opacity-80">
                {type === 'exporter'
                  ? 'I supply products to international markets'
                  : 'I source products from global suppliers'}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Full Name */}
      <div className="space-y-2">
        <Label htmlFor="fullName" className="text-foreground">
          Full name <span className="text-destructive">*</span>
        </Label>
        <Input
          id="fullName"
          placeholder="John Smith"
          value={profile.fullName}
          onChange={(e) => onChange({ ...profile, fullName: e.target.value })}
          className="bg-background border-border text-foreground placeholder:text-muted-foreground"
        />
      </div>

      {/* Phone */}
      <div className="space-y-2">
        <Label htmlFor="phone" className="text-foreground">
          Phone number <span className="text-destructive">*</span>
        </Label>
        <Input
          id="phone"
          type="tel"
          placeholder="+1 (555) 123-4567"
          value={profile.phone}
          onChange={(e) => onChange({ ...profile, phone: e.target.value })}
          className="bg-background border-border text-foreground placeholder:text-muted-foreground"
        />
      </div>
    </>
  )
}

/* ─── Step 2: Organization ─── */

function StepOrganization({
  org,
  onChange,
}: {
  org: OrgForm
  onChange: (o: OrgForm) => void
}) {
  return (
    <>
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
          <Building2 className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-foreground">Organization Details</h2>
          <p className="text-sm text-muted-foreground">
            Provide your company information for trade matching
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="space-y-2 sm:col-span-2">
          <Label htmlFor="companyName" className="text-foreground">
            Company name <span className="text-destructive">*</span>
          </Label>
          <Input
            id="companyName"
            placeholder="Acme Trading Co."
            value={org.companyName}
            onChange={(e) => onChange({ ...org, companyName: e.target.value })}
            className="bg-background border-border text-foreground placeholder:text-muted-foreground"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="regNumber" className="text-foreground">
            Registration number
          </Label>
          <Input
            id="regNumber"
            placeholder="Optional"
            value={org.registrationNumber}
            onChange={(e) => onChange({ ...org, registrationNumber: e.target.value })}
            className="bg-background border-border text-foreground placeholder:text-muted-foreground"
          />
        </div>

        <div className="space-y-2">
          <Label className="text-foreground">
            Country <span className="text-destructive">*</span>
          </Label>
          <Select
            value={org.country}
            onValueChange={(v) => onChange({ ...org, country: v })}
          >
            <SelectTrigger className="bg-background border-border text-foreground">
              <SelectValue placeholder="Select country" />
            </SelectTrigger>
            <SelectContent className="bg-card border-border">
              {COUNTRIES.map((c) => (
                <SelectItem key={c} value={c} className="text-foreground">
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="city" className="text-foreground">City</Label>
          <Input
            id="city"
            placeholder="New York"
            value={org.city}
            onChange={(e) => onChange({ ...org, city: e.target.value })}
            className="bg-background border-border text-foreground placeholder:text-muted-foreground"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="website" className="text-foreground">Website</Label>
          <Input
            id="website"
            type="url"
            placeholder="https://example.com"
            value={org.website}
            onChange={(e) => onChange({ ...org, website: e.target.value })}
            className="bg-background border-border text-foreground placeholder:text-muted-foreground"
          />
        </div>

        <div className="space-y-2 sm:col-span-2">
          <Label htmlFor="address" className="text-foreground">Address</Label>
          <Input
            id="address"
            placeholder="123 Trade Street"
            value={org.address}
            onChange={(e) => onChange({ ...org, address: e.target.value })}
            className="bg-background border-border text-foreground placeholder:text-muted-foreground"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="yearEst" className="text-foreground">Year established</Label>
          <Input
            id="yearEst"
            type="number"
            placeholder="2015"
            value={org.yearEstablished}
            onChange={(e) => onChange({ ...org, yearEstablished: e.target.value })}
            className="bg-background border-border text-foreground placeholder:text-muted-foreground"
          />
        </div>

        <div className="space-y-2">
          <Label className="text-foreground">Number of employees</Label>
          <Select
            value={org.numberOfEmployees}
            onValueChange={(v) => onChange({ ...org, numberOfEmployees: v })}
          >
            <SelectTrigger className="bg-background border-border text-foreground">
              <SelectValue placeholder="Select range" />
            </SelectTrigger>
            <SelectContent className="bg-card border-border">
              {['1-10', '11-50', '51-200', '201-500', '500+'].map((r) => (
                <SelectItem key={r} value={r} className="text-foreground">
                  {r}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label className="text-foreground">Annual revenue (USD)</Label>
          <Select
            value={org.annualRevenue}
            onValueChange={(v) => onChange({ ...org, annualRevenue: v })}
          >
            <SelectTrigger className="bg-background border-border text-foreground">
              <SelectValue placeholder="Select range" />
            </SelectTrigger>
            <SelectContent className="bg-card border-border">
              {[
                'Under $100K', '$100K-$500K', '$500K-$1M', '$1M-$5M',
                '$5M-$25M', '$25M-$100M', '$100M+',
              ].map((r) => (
                <SelectItem key={r} value={r} className="text-foreground">
                  {r}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2 sm:col-span-2">
          <Label htmlFor="orgDesc" className="text-foreground">Company description</Label>
          <Textarea
            id="orgDesc"
            placeholder="Brief description of your company and what you trade..."
            value={org.description}
            onChange={(e) => onChange({ ...org, description: e.target.value })}
            className="bg-background border-border text-foreground placeholder:text-muted-foreground resize-none"
            rows={3}
          />
        </div>
      </div>
    </>
  )
}

/* ─── Step 3: Trade Info ─── */

function StepTradeInfo({
  trade,
  onChange,
  toggleArrayItem,
}: {
  trade: TradeForm
  onChange: (t: TradeForm) => void
  toggleArrayItem: (arr: string[], item: string) => string[]
}) {
  return (
    <>
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
          <Package className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-foreground">Trade Information</h2>
          <p className="text-sm text-muted-foreground">
            Configure your products, markets, and trade preferences
          </p>
        </div>
      </div>

      {/* Product Categories */}
      <div className="space-y-2">
        <Label className="text-foreground">
          Product categories <span className="text-destructive">*</span>
        </Label>
        <p className="text-xs text-muted-foreground">Select all that apply</p>
        <div className="flex flex-wrap gap-2">
          {PRODUCT_CATEGORIES.map((cat) => (
            <button
              key={cat}
              type="button"
              onClick={() =>
                onChange({
                  ...trade,
                  productCategories: toggleArrayItem(trade.productCategories, cat),
                })
              }
              className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                trade.productCategories.includes(cat)
                  ? 'bg-primary/20 border-primary text-primary'
                  : 'bg-background border-border text-muted-foreground hover:border-muted-foreground'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Primary Markets */}
      <div className="space-y-2">
        <Label className="text-foreground">
          Primary markets <span className="text-destructive">*</span>
        </Label>
        <p className="text-xs text-muted-foreground">Countries you trade with</p>
        <div className="flex flex-wrap gap-2">
          {COUNTRIES.slice(0, -1).map((c) => (
            <button
              key={c}
              type="button"
              onClick={() =>
                onChange({
                  ...trade,
                  primaryMarkets: toggleArrayItem(trade.primaryMarkets, c),
                })
              }
              className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                trade.primaryMarkets.includes(c)
                  ? 'bg-primary/20 border-primary text-primary'
                  : 'bg-background border-border text-muted-foreground hover:border-muted-foreground'
              }`}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      {/* Certifications */}
      <div className="space-y-2">
        <Label className="text-foreground">Certifications</Label>
        <div className="flex flex-wrap gap-2">
          {CERTIFICATIONS.map((cert) => (
            <button
              key={cert}
              type="button"
              onClick={() =>
                onChange({
                  ...trade,
                  certifications: toggleArrayItem(trade.certifications, cert),
                })
              }
              className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                trade.certifications.includes(cert)
                  ? 'bg-primary/20 border-primary text-primary'
                  : 'bg-background border-border text-muted-foreground hover:border-muted-foreground'
              }`}
            >
              {cert}
            </button>
          ))}
        </div>
      </div>

      {/* Min Order Value */}
      <div className="space-y-2">
        <Label htmlFor="minOrder" className="text-foreground">Minimum order value (USD)</Label>
        <Input
          id="minOrder"
          placeholder="e.g. $5,000"
          value={trade.minOrderValue}
          onChange={(e) => onChange({ ...trade, minOrderValue: e.target.value })}
          className="bg-background border-border text-foreground placeholder:text-muted-foreground"
        />
      </div>

      {/* Payment Terms */}
      <div className="space-y-2">
        <Label className="text-foreground">Preferred payment terms</Label>
        <div className="flex flex-wrap gap-2">
          {PAYMENT_TERMS.map((term) => (
            <button
              key={term}
              type="button"
              onClick={() =>
                onChange({
                  ...trade,
                  preferredPaymentTerms: toggleArrayItem(trade.preferredPaymentTerms, term),
                })
              }
              className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                trade.preferredPaymentTerms.includes(term)
                  ? 'bg-primary/20 border-primary text-primary'
                  : 'bg-background border-border text-muted-foreground hover:border-muted-foreground'
              }`}
            >
              {term}
            </button>
          ))}
        </div>
      </div>

      {/* Incoterms */}
      <div className="space-y-2">
        <Label className="text-foreground">Preferred Incoterms</Label>
        <div className="flex flex-wrap gap-2">
          {INCOTERMS.map((term) => (
            <button
              key={term}
              type="button"
              onClick={() =>
                onChange({
                  ...trade,
                  preferredIncoterms: toggleArrayItem(trade.preferredIncoterms, term),
                })
              }
              className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                trade.preferredIncoterms.includes(term)
                  ? 'bg-primary/20 border-primary text-primary'
                  : 'bg-background border-border text-muted-foreground hover:border-muted-foreground'
              }`}
            >
              {term}
            </button>
          ))}
        </div>
      </div>

      {/* Logistics */}
      <div className="space-y-2">
        <Label className="text-foreground">Logistics capabilities</Label>
        <div className="flex flex-wrap gap-2">
          {LOGISTICS.map((cap) => (
            <button
              key={cap}
              type="button"
              onClick={() =>
                onChange({
                  ...trade,
                  logisticsCapabilities: toggleArrayItem(trade.logisticsCapabilities, cap),
                })
              }
              className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                trade.logisticsCapabilities.includes(cap)
                  ? 'bg-primary/20 border-primary text-primary'
                  : 'bg-background border-border text-muted-foreground hover:border-muted-foreground'
              }`}
            >
              {cap}
            </button>
          ))}
        </div>
      </div>
    </>
  )
}

/* ─── Step 4: KYC Documents ─── */

function StepDocuments({
  files,
  onFilesChange,
}: {
  files: UploadedFile[]
  onFilesChange: (f: UploadedFile[]) => void
}) {
  return (
    <>
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
          <ShieldCheck className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-foreground">Verification Documents</h2>
          <p className="text-sm text-muted-foreground">
            Upload documents to verify your business (optional, can be done later)
          </p>
        </div>
      </div>

      <div className="bg-background border border-border rounded-lg p-4 space-y-2">
        <p className="text-sm text-foreground font-medium">Recommended documents:</p>
        <ul className="text-xs text-muted-foreground space-y-1 list-disc list-inside">
          <li>Business license / Company registration</li>
          <li>Tax certificate / VAT / GST registration</li>
        </ul>
      </div>

      <FileUploadZone
        files={files}
        onFilesChange={onFilesChange}
        maxFiles={10}
        maxSizeMB={10}
        label="Upload your documents"
        description="PDF, JPG, or PNG files up to 10 MB each. Maximum 10 files."
      />

      <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
        <p className="text-xs text-muted-foreground">
          Documents are securely stored and encrypted. They will be reviewed by our verification team to establish your trust score. You can skip this step and upload documents later from your profile settings.
        </p>
      </div>
    </>
  )
}
