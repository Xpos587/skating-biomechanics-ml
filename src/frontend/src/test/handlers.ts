import { HttpResponse, http } from "msw"

// Mock API handlers for /api/detect and /api/process
export const handlers = [
  // POST /api/detect
  http.post("/api/detect", async ({ request }) => {
    const formData = await request.formData()
    const file = formData.get("video") as File

    if (!file) {
      return HttpResponse.json({ detail: "No video file provided" }, { status: 400 })
    }

    // Mock response
    return HttpResponse.json({
      persons: [
        {
          track_id: 1,
          hits: 100,
          bbox: [0.3, 0.2, 0.7, 0.8],
          mid_hip: [0.5, 0.5],
        },
      ],
      preview_image: "data:image/png;base64,mock",
      video_path: "/tmp/mock_video.mp4",
      auto_click: null,
      status: "success",
    })
  }),

  // POST /api/process (SSE - mock the initial request)
  http.post("/api/process", async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>

    if (!body.video_path || !body.person_click) {
      return HttpResponse.json({ detail: "Missing required fields" }, { status: 400 })
    }

    // Note: SSE is handled in src/test/server.ts
    return HttpResponse.json({ status: "processing" })
  }),
]
