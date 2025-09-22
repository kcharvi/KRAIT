import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    console.log('GPU Execution Request:', {
      kernel_code: body.kernel_code?.substring(0, 100) + '...',
      hardware: body.hardware,
      provider: body.provider,
      timestamp: new Date().toISOString()
    })
    
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'
    const endpoint = `${backendUrl}/api/v1/gpu/execute-kernel`
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })


    if (!response.ok) {
      const errorText = await response.text()
      console.error('❌ Backend error response:', errorText)
      
      if (errorText.includes('<!DOCTYPE') || errorText.includes('<html')) {
        return NextResponse.json(
          { 
            error: 'Backend server not running or endpoint not found',
            details: `Backend at ${backendUrl} returned HTML instead of JSON`,
            status: response.status,
            suggestion: 'Please ensure the backend is running on port 8000'
          },
          { status: 503 }
        )
      }
      
      return NextResponse.json(
        { 
          error: `Backend responded with ${response.status}`,
          details: errorText,
          status: response.status
        },
        { status: response.status }
      )
    }

    const data = await response.json()
    console.log('GPU execution successful:', data)
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('GPU execution error:', error)
    
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { 
          error: 'Cannot connect to backend server',
          details: error.message,
          suggestion: 'Please ensure the backend is running on http://localhost:8000'
        },
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { 
        error: 'Failed to process GPU execution request',
        details: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    )
  }
}

