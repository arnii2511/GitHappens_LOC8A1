import Link from 'next/link'
import { Globe, Mail } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function SignUpSuccessPage() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-8">
        {/* Logo */}
        <div className="flex flex-col items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-cyan-500 rounded-lg blur opacity-50" />
            <div className="relative bg-card px-3 py-2 rounded-lg flex items-center gap-2">
              <Globe className="w-6 h-6 text-secondary" />
              <span className="font-bold text-lg bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                NexPort
              </span>
            </div>
          </div>
        </div>

        {/* Success card */}
        <div className="bg-card border border-border rounded-xl p-8 text-center space-y-5">
          <div className="mx-auto w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center">
            <Mail className="w-7 h-7 text-primary" />
          </div>

          <div className="space-y-2">
            <h1 className="text-xl font-semibold text-foreground">Check your email</h1>
            <p className="text-muted-foreground text-sm leading-relaxed">
              {"We've sent a confirmation link to your email address. Click the link to verify your account and continue setting up your profile."}
            </p>
          </div>

          <div className="bg-background border border-border rounded-lg p-4 text-left space-y-2">
            <p className="text-sm text-foreground font-medium">What happens next?</p>
            <ol className="text-xs text-muted-foreground space-y-1.5 list-decimal list-inside">
              <li>Open the confirmation email</li>
              <li>Click the verification link</li>
              <li>{"You'll be redirected to complete your profile setup"}</li>
            </ol>
          </div>

          <Button asChild variant="outline" className="w-full border-border text-foreground hover:bg-background">
            <Link href="/auth/login">Back to sign in</Link>
          </Button>
        </div>
      </div>
    </div>
  )
}
