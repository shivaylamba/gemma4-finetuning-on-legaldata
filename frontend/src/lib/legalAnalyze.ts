export type LegalAnalyzeRequest = {
  query: string
  max_tokens?: number
  temperature?: number
}

export type LegalAnalyzeResponse = {
  answer: string
  model: string
  usage: {
    completion_tokens: number
    prompt_tokens: number
    total_tokens: number
    completion_tokens_details: null
    prompt_tokens_details: null
  }
}

const MOCK_ANSWERS = [
  'This is a placeholder analysis. In practice, the relevant statutory text would be retrieved and cited alongside any jurisdictional caveats.',
  'Under the statutory framework you are exploring, courts typically look at the plain meaning of the provision, the legislative intent, and any applicable regulations.',
]

function randomMockAnswer(): string {
  const i = Math.floor(Math.random() * MOCK_ANSWERS.length)
  return MOCK_ANSWERS[i]
}

function fakeUsage(query: string): LegalAnalyzeResponse['usage'] {
  const prompt_tokens = Math.min(256, 32 + Math.ceil(query.length / 4))
  const completion_tokens = 40 + Math.floor(Math.random() * 120)
  return {
    completion_tokens,
    prompt_tokens,
    total_tokens: prompt_tokens + completion_tokens,
    completion_tokens_details: null,
    prompt_tokens_details: null,
  }
}

/** Only `true` enables mock responses (local UI testing). */
export const USE_MOCK_LEGAL_ANALYZE =
  import.meta.env.VITE_USE_MOCK_LEGAL_ANALYZE === 'true'

/** Full URL to POST `/v1/legal/analyze` (override via env for other deployments). */
export const DEFAULT_LEGAL_ANALYZE_URL =
  'https://carmen-monster-military-permissions.trycloudflare.com/v1/legal/analyze'

function resolveUrl(): string {
  const u = import.meta.env.VITE_LEGAL_ANALYZE_URL as string | undefined
  if (u?.trim()) return u.trim()
  return DEFAULT_LEGAL_ANALYZE_URL
}

export type LegalAnalyzeOptions = {
  /** Called with growing answer text when the server streams (SSE or chunked text). */
  onDelta?: (accumulated: string) => void
}

function extractDeltaFromJsonObject(obj: unknown): string | null {
  if (!obj || typeof obj !== 'object') return null
  const o = obj as Record<string, unknown>
  if (typeof o.answer === 'string') return o.answer
  const choices = o.choices
  if (Array.isArray(choices) && choices[0] && typeof choices[0] === 'object') {
    const c0 = choices[0] as Record<string, unknown>
    const delta = c0.delta
    if (delta && typeof delta === 'object') {
      const d = delta as Record<string, unknown>
      if (typeof d.content === 'string') return d.content
    }
  }
  if (typeof o.token === 'string') return o.token
  if (typeof o.text === 'string') return o.text
  return null
}

async function parseSseOrTextStream(
  res: Response,
  onDelta: (text: string) => void,
): Promise<LegalAnalyzeResponse> {
  const body = res.body
  if (!body) throw new Error('Empty response body')

  const reader = body.getReader()
  const decoder = new TextDecoder()
  let lineBuf = ''
  let textAcc = ''
  let lastMeta: Partial<LegalAnalyzeResponse> = {}

  const applyObject = (obj: unknown) => {
    if (!obj || typeof obj !== 'object') return
    const o = obj as Record<string, unknown>
    if (typeof o.answer === 'string') {
      textAcc = o.answer
      onDelta(textAcc)
    }
    const piece = extractDeltaFromJsonObject(obj)
    if (piece !== null && typeof o.answer !== 'string') {
      textAcc += piece
      onDelta(textAcc)
    }
    if (typeof o.model === 'string') lastMeta.model = o.model
    if (o.usage && typeof o.usage === 'object') {
      lastMeta.usage = o.usage as LegalAnalyzeResponse['usage']
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    lineBuf += decoder.decode(value, { stream: true })
    const lines = lineBuf.split(/\r?\n/)
    lineBuf = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith(':')) continue

      if (trimmed.startsWith('data:')) {
        const raw = trimmed.slice(5).trim()
        if (raw === '[DONE]') continue
        try {
          applyObject(JSON.parse(raw))
        } catch {
          // ignore partial / non-JSON data lines
        }
        continue
      }

      try {
        applyObject(JSON.parse(trimmed))
      } catch {
        // plain text line fragment
        textAcc += trimmed
        onDelta(textAcc)
      }
    }
  }

  const tail = lineBuf.trim()
  if (tail) {
    if (tail.startsWith('data:')) {
      const raw = tail.slice(5).trim()
      if (raw && raw !== '[DONE]') {
        try {
          applyObject(JSON.parse(raw))
        } catch {
          /* noop */
        }
      }
    } else {
      try {
        const parsed = JSON.parse(tail) as LegalAnalyzeResponse
        if (typeof parsed.answer === 'string') {
          textAcc = parsed.answer
          onDelta(textAcc)
          lastMeta = { ...parsed }
        }
      } catch {
        textAcc += tail
        onDelta(textAcc)
      }
    }
  }

  if (!textAcc) {
    throw new Error('Stream ended without answer text')
  }

  return {
    answer: textAcc,
    model: (lastMeta.model as string) ?? 'legal-lora',
    usage:
      lastMeta.usage ??
      ({
        completion_tokens: 0,
        prompt_tokens: 0,
        total_tokens: 0,
        completion_tokens_details: null,
        prompt_tokens_details: null,
      } as LegalAnalyzeResponse['usage']),
  }
}

export async function legalAnalyze(
  body: LegalAnalyzeRequest,
  options?: LegalAnalyzeOptions,
): Promise<LegalAnalyzeResponse> {
  if (USE_MOCK_LEGAL_ANALYZE) {
    await new Promise((r) => setTimeout(r, 400 + Math.random() * 400))
    const answer = randomMockAnswer()
    options?.onDelta?.(answer)
    return {
      answer,
      model: 'legal-lora',
      usage: fakeUsage(body.query),
    }
  }

  const url = resolveUrl()
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json, text/event-stream',
    },
    body: JSON.stringify({
      query: body.query,
      max_tokens: body.max_tokens ?? 512,
      temperature: body.temperature ?? 0.3,
    }),
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Request failed (${res.status})`)
  }

  const ct = res.headers.get('content-type') ?? ''

  if (ct.includes('application/json')) {
    const data = (await res.json()) as LegalAnalyzeResponse
    if (typeof data.answer !== 'string') {
      throw new Error('Invalid response: missing answer')
    }
    options?.onDelta?.(data.answer)
    return data
  }

  if (ct.includes('text/event-stream') || ct.includes('text/plain')) {
    const onDelta = options?.onDelta
    if (!onDelta) {
      return parseSseOrTextStream(res, () => {})
    }
    return parseSseOrTextStream(res, onDelta)
  }

  const buf = await res.arrayBuffer()
  const text = new TextDecoder().decode(buf)
  try {
    const data = JSON.parse(text) as LegalAnalyzeResponse
    if (typeof data.answer === 'string') {
      options?.onDelta?.(data.answer)
      return data
    }
  } catch {
    /* fall through */
  }
  throw new Error('Unexpected response format from legal analyze endpoint')
}
