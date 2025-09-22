import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'
    const endpoint = `${backendUrl}/api/v1/gpu/status`
    
    const response = await fetch(endpoint, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('❌ Backend status error:', errorText)
      
      return NextResponse.json({
        status: 'backend_unavailable',
        provider: 'unknown',
        error: 'Backend server not responding',
        suggestion: 'Please start the backend server'
      })
    }

    const data = await response.json()
    console.log('GPU status retrieved:', data)
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('GPU status error:', error)
    
    return NextResponse.json({
      status: 'error',
      provider: 'unknown',
      error: error instanceof Error ? error.message : 'Unknown error',
      suggestion: 'Please check backend connection'
    })
  }
}

