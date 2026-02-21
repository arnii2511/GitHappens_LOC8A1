import Link from 'next/link'
import { Globe, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function AuthErrorPage() {
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

        {/* Error card */}
        <div className="bg-card border border-border rounded-xl p-8 text-center space-y-5">
          <div className="mx-auto w-14 h-14 rounded-full bg-destructive/10 flex items-center justify-center">
            <AlertTriangle className="w-7 h-7 text-destructive" />
          </div>

          <div className="space-y-2">
            <h1 className="text-xl font-semibold text-foreground">Authentication Error</h1>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Something went wrong during authentication. This could be an expired link, an invalid token, or a temporary issue. Please try again.
            </p>
          </div>

          <div className="flex flex-col gap-3">
            <Button asChild className="w-full bg-primary hover:bg-primary/90 text-primary-foreground">
              <Link href="/auth/login">Back to sign in</Link>
            </Button>
            <Button asChild variant="outline" className="w-full border-border text-foreground hover:bg-background">
              <Link href="/auth/sign-up">Create a new account</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
