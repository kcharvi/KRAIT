import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    console.log('🔍 Checking GPU status...')
    
    // Forward the request to your backend
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'
    const endpoint = `${backendUrl}/api/v1/gpu/status`
    
    console.log('📡 Calling backend status:', endpoint)
    
    const response = await fetch(endpoint, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    console.log('📊 Backend status response:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('❌ Backend status error:', errorText)
      
      // Return mock status if backend is down
      return NextResponse.json({
        status: 'backend_unavailable',
        provider: 'unknown',
        error: 'Backend server not responding',
        suggestion: 'Please start the backend server'
      })
    }

    const data = await response.json()
    console.log('✅ GPU status retrieved:', data)
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('💥 GPU status error:', error)
    
    // Return mock status on error
    return NextResponse.json({
      status: 'error',
      provider: 'unknown',
      error: error instanceof Error ? error.message : 'Unknown error',
      suggestion: 'Please check backend connection'
    })
  }
}

