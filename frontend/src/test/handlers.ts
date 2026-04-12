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
      video_key: "input/mock-video.mp4",
      auto_click: null,
      status: "success",
    })
  }),
]
