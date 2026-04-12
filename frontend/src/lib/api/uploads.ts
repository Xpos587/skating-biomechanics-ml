// src/frontend/src/lib/api/uploads.ts

import { z } from "zod"
import { apiFetch } from "@/lib/api-client"

const InitResponseSchema = z.object({
  upload_id: z.string(),
  key: z.string(),
  chunk_size: z.number(),
  part_count: z.number(),
  parts: z.array(z.object({ part_number: z.number(), url: z.string() })),
})

export class ChunkedUploader {
  private file: File
  private onProgress: (loaded: number, total: number) => void

  constructor(file: File, onProgress: (loaded: number, total: number) => void) {
    this.file = file
    this.onProgress = onProgress
  }

  async upload(): Promise<string> {
    // Init multipart upload
    const init = await apiFetch(
      `/uploads/init?file_name=${encodeURIComponent(this.file.name)}&content_type=${this.file.type}&total_size=${this.file.size}`,
      InitResponseSchema,
      { method: "POST" },
    )

    const CHUNK_SIZE = init.chunk_size
    const CONCURRENCY = 3
    let uploaded = 0

    // Upload parts with limited concurrency
    const queue = [...init.parts]
    const inFlight = new Set<Promise<void>>()

    const processPart = async (part: { part_number: number; url: string }) => {
      const start = (part.part_number - 1) * CHUNK_SIZE
      const end = Math.min(start + CHUNK_SIZE, this.file.size)
      const chunk = this.file.slice(start, end)

      const res = await fetch(part.url, { method: "PUT", body: chunk })
      if (!res.ok) throw new Error(`Part ${part.part_number} upload failed`)

      uploaded += end - start
      this.onProgress(uploaded, this.file.size)
    }

    while (queue.length > 0 || inFlight.size > 0) {
      while (inFlight.size < CONCURRENCY && queue.length > 0) {
        const part = queue.shift()
        if (!part) break
        const promise = processPart(part).then(() => {
          inFlight.delete(promise)
        })
        inFlight.add(promise)
      }
      if (inFlight.size > 0) {
        await Promise.race(inFlight)
      }
    }

    // Complete upload
    await apiFetch(`/uploads/${init.upload_id}/complete`, z.object({ status: z.string() }), {
      method: "POST",
    })

    return init.key
  }
}
