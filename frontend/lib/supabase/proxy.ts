import { createServerClient, type CookieOptions } from "@supabase/ssr"
import { NextRequest, NextResponse } from "next/server"

function clearStaleAuthCookies(request: NextRequest, response: NextResponse) {
  const staleAuthCookies = request.cookies
    .getAll()
    .filter(({ name }) => name.startsWith('sb-') && name.includes('auth-token'))

  staleAuthCookies.forEach(({ name }) => {
    response.cookies.set(name, '', { path: '/', maxAge: 0 })
  })
}

export async function updateSession(request: NextRequest) {
  let supabaseResponse = NextResponse.next({ request })
  const { pathname } = request.nextUrl
  const bypassAuth = process.env.DEV_BYPASS_AUTH === 'true'

  // Dev convenience: bypass auth middleware protection when explicitly enabled.
  if (bypassAuth) {
    return supabaseResponse
  }

  // Public auth routes should not run session refresh logic.
  const publicPaths = [
    '/auth/login',
    '/auth/sign-up',
    '/auth/sign-up-success',
    '/auth/error'
  ]
  const isPublicAuth = publicPaths.some((p) => pathname.startsWith(p))
  if (isPublicAuth) {
    return supabaseResponse
  }

  type CookieToSet = {
    name: string
    value: string
    options: CookieOptions
  }

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!supabaseUrl || !supabaseAnonKey) {
    return supabaseResponse
  }

  const supabase = createServerClient(
    supabaseUrl,
    supabaseAnonKey,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet: CookieToSet[]) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          )
          supabaseResponse = NextResponse.next({ request })
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  // 🔥 ONLY NOW check user
  let user = null
  const hasSupabaseAuthCookie = request.cookies
    .getAll()
    .some(({ name }) => name.startsWith('sb-') && name.includes('auth-token'))

  if (hasSupabaseAuthCookie) {
    try {
      const { data, error } = await supabase.auth.getUser()
      user = data?.user ?? null
      if (error) {
        const code = (error as { code?: string }).code
        if (code === 'refresh_token_not_found') {
          clearStaleAuthCookies(request, supabaseResponse)
        } else {
          console.error('Supabase auth error in proxy:', error)
        }
      }
    } catch (error) {
      const code = (error as { code?: string }).code
      if (code === 'refresh_token_not_found') {
        clearStaleAuthCookies(request, supabaseResponse)
      } else {
        console.error('Supabase auth error in proxy:', error)
      }
    }
  }

  // Protected: onboarding
  if (pathname.startsWith('/auth/onboarding')) {
    if (!user) {
      const url = request.nextUrl.clone()
      url.pathname = '/auth/login'
      return NextResponse.redirect(url)
    }
    return supabaseResponse
  }

  // Protected: profile
  if (pathname.startsWith('/profile')) {
    if (!user) {
      const url = request.nextUrl.clone()
      url.pathname = '/auth/login'
      return NextResponse.redirect(url)
    }
    return supabaseResponse
  }

  // Root protection
  if (!user && pathname === '/') {
    const url = request.nextUrl.clone()
    url.pathname = '/auth/login'
    return NextResponse.redirect(url)
  }

  return supabaseResponse
}
