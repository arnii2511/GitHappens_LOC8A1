import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'

export async function updateSession(request: NextRequest) {
  let supabaseResponse = NextResponse.next({
    request,
  })

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!supabaseUrl || !supabaseAnonKey) {
    // Env vars not yet available -- let the request through so pages can render
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
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value),
          )
          supabaseResponse = NextResponse.next({
            request,
          })
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options),
          )
        },
      },
    },
  )

  const {
    data: { user },
  } = await supabase.auth.getUser()

  const { pathname } = request.nextUrl

  // Public auth routes - allow freely
  const publicPaths = ['/auth/login', '/auth/sign-up', '/auth/sign-up-success', '/auth/error']
  const isPublicAuth = publicPaths.some((p) => pathname.startsWith(p))

  if (isPublicAuth) {
    // If user is already logged in and tries to access auth pages, redirect to main app
    if (user) {
      // Check if they have completed onboarding by checking profile
      const { data: profile } = await supabase
        .from('profiles')
        .select('full_name')
        .eq('id', user.id)
        .single()

      const url = request.nextUrl.clone()
      if (profile?.full_name) {
        url.pathname = '/'
        return NextResponse.redirect(url)
      } else {
        url.pathname = '/auth/onboarding'
        return NextResponse.redirect(url)
      }
    }
    return supabaseResponse
  }

  // Protected: onboarding route
  if (pathname.startsWith('/auth/onboarding')) {
    if (!user) {
      const url = request.nextUrl.clone()
      url.pathname = '/auth/login'
      return NextResponse.redirect(url)
    }
    return supabaseResponse
  }

  // Protected: profile page
  if (pathname.startsWith('/profile')) {
    if (!user) {
      const url = request.nextUrl.clone()
      url.pathname = '/auth/login'
      return NextResponse.redirect(url)
    }
    return supabaseResponse
  }

  // Protected: API routes
  if (pathname.startsWith('/api/profile')) {
    // Let the API route itself handle auth (it checks getUser)
    return supabaseResponse
  }

  // Protected: main app (everything else that isn't a public path)
  if (!user && pathname === '/') {
    const url = request.nextUrl.clone()
    url.pathname = '/auth/login'
    return NextResponse.redirect(url)
  }

  return supabaseResponse
}
