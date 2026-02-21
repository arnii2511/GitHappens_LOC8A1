"use client"

import Link from "next/link"

export default function Page() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">

      {/* Header */}
      <div className="flex items-center gap-3 p-6">
        <div className="w-10 h-10 bg-primary rounded-lg"></div>
        <h1 className="text-2xl font-semibold">NexPort</h1>
      </div>

      {/* Hero Section */}
      <div className="flex flex-col items-center justify-center flex-1 px-6 text-center max-w-3xl mx-auto">

        <h2 className="text-4xl md:text-5xl font-bold mb-6 leading-tight">
          AI-Powered Export Growth Engine
        </h2>

        <p className="text-lg md:text-xl text-muted-foreground mb-12">
          Scale your manufacturing exports with intelligent lead generation
          and automated outreach
        </p>

        {/* Steps */}
        <div className="space-y-8 text-left w-full max-w-xl mb-14">

          <div className="flex gap-4 items-start">
            <div className="w-10 h-10 flex items-center justify-center bg-card border border-border rounded-lg font-bold">
              1
            </div>
            <div>
              <h3 className="font-semibold text-lg">Approve Curated Leads</h3>
              <p className="text-muted-foreground text-sm">
                Swipe through AI-selected export opportunities tailored to your ICP
              </p>
            </div>
          </div>

          <div className="flex gap-4 items-start">
            <div className="w-10 h-10 flex items-center justify-center bg-card border border-border rounded-lg font-bold">
              2
            </div>
            <div>
              <h3 className="font-semibold text-lg">AI Handles Outreach</h3>
              <p className="text-muted-foreground text-sm">
                Omni-channel automation across LinkedIn, Email, WhatsApp, and Calls
              </p>
            </div>
          </div>

          <div className="flex gap-4 items-start">
            <div className="w-10 h-10 flex items-center justify-center bg-card border border-border rounded-lg font-bold">
              3
            </div>
            <div>
              <h3 className="font-semibold text-lg">Attend Scheduled Meetings</h3>
              <p className="text-muted-foreground text-sm">
                Focus on closing deals while AI manages the pipeline
              </p>
            </div>
          </div>

        </div>

        {/* CTA Buttons */}
        <div className="flex gap-6">
          <Link
            href="/auth/login"
            className="px-6 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:opacity-90 transition"
          >
            Login
          </Link>

          <Link
            href="/auth/sign-up"
            className="px-6 py-3 border border-border rounded-lg font-semibold hover:bg-card transition"
          >
            Sign Up
          </Link>
        </div>

      </div>
    </div>
  )
}