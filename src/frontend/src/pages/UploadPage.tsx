import { AlertCircle, CheckCircle, Loader2, Upload } from "lucide-react"
import { useCallback, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { detectPersons } from "@/lib/api"
import type { DetectResponse, PersonClick } from "@/types"

type Status = "idle" | "uploading" | "detecting" | "ready" | "error"

export default function UploadPage() {
  const navigate = useNavigate()
  const fileRef = useRef<HTMLInputElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<Status>("idle")
  const [error, setError] = useState("")
  const [detectResult, setDetectResult] = useState<DetectResponse | null>(null)
  const [selectedPerson, setSelectedPerson] = useState<number | null>(null)
  const [clickCoord, setClickCoord] = useState<PersonClick | null>(null)

  // Settings
  const [frameSkip, setFrameSkip] = useState(1)
  const [layer, setLayer] = useState(3)
  const [tracking, setTracking] = useState("auto")
  const [doExport, setDoExport] = useState(true)

  // Derived: selected person's bbox as CSS percentage
  const selectedPersonData =
    detectResult && selectedPerson !== null
      ? detectResult.persons.find(p => p.track_id === selectedPerson)
      : undefined

  const handleFile = useCallback(
    async (f: File) => {
      setFile(f)
      setStatus("uploading")
      setError("")
      setDetectResult(null)
      setSelectedPerson(null)
      setClickCoord(null)

      const { data, error: err } = await detectPersons(f, tracking)
      if (err) {
        setError(err)
        setStatus("error")
        return
      }

      const resp = data as DetectResponse
      setDetectResult(resp)

      if (resp.auto_click) {
        setClickCoord(resp.auto_click)
        setSelectedPerson(resp.persons[0]?.track_id ?? 0)
      }

      setStatus("ready")
    },
    [tracking],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const f = e.dataTransfer.files[0]
      if (f) handleFile(f)
    },
    [handleFile],
  )

  const pickPersonAt = useCallback(
    (clientX: number, clientY: number) => {
      if (!detectResult || !imgRef.current) return

      const rect = imgRef.current.getBoundingClientRect()
      const scaleX = imgRef.current.naturalWidth / rect.width
      const scaleY = imgRef.current.naturalHeight / rect.height
      const px = Math.round((clientX - rect.left) * scaleX)
      const py = Math.round((clientY - rect.top) * scaleY)

      let bestDist = Infinity
      let bestId = 0
      const natW = imgRef.current.naturalWidth
      const natH = imgRef.current.naturalHeight
      for (const p of detectResult.persons) {
        const hx = p.mid_hip[0] * natW
        const hy = p.mid_hip[1] * natH
        const dist = Math.hypot(px - hx, py - hy)
        if (dist < bestDist) {
          bestDist = dist
          bestId = p.track_id
        }
      }

      setClickCoord({ x: px, y: py })
      setSelectedPerson(bestId)
    },
    [detectResult],
  )

  const pickPerson = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      pickPersonAt(e.clientX, e.clientY)
    },
    [pickPersonAt],
  )

  const selectFromList = useCallback(
    (trackId: number) => {
      setSelectedPerson(trackId)
      const person = detectResult?.persons.find(p => p.track_id === trackId)
      if (person && imgRef.current) {
        const px = Math.round(person.mid_hip[0] * imgRef.current.naturalWidth)
        const py = Math.round(person.mid_hip[1] * imgRef.current.naturalHeight)
        setClickCoord({ x: px, y: py })
      }
    },
    [detectResult],
  )

  const handleAnalyze = () => {
    if (!detectResult || !clickCoord) return
    const params = new URLSearchParams({
      video_path: detectResult.video_path,
      person_click: `${clickCoord.x},${clickCoord.y}`,
      frame_skip: String(frameSkip),
      layer: String(layer),
      tracking,
      export: String(doExport),
    })
    navigate(`/analyze?${params.toString()}`)
  }

  const isAnalyzing = status === "uploading" || status === "detecting"
  const previewSrc = detectResult ? `data:image/png;base64,${detectResult.preview_image}` : ""

  // Compute bbox overlay CSS
  const bboxStyle = selectedPersonData
    ? {
        left: `${selectedPersonData.bbox[0] * 100}%`,
        top: `${selectedPersonData.bbox[1] * 100}%`,
        width: `${(selectedPersonData.bbox[2] - selectedPersonData.bbox[0]) * 100}%`,
        height: `${(selectedPersonData.bbox[3] - selectedPersonData.bbox[1]) * 100}%`,
      }
    : undefined

  return (
    <div className="mx-auto max-w-5xl p-6">
      {/* Upload zone */}
      {status === "idle" && (
        /* biome-ignore lint/a11y/useSemanticElements: div needed for drag-and-drop */
        <div
          role="button"
          tabIndex={0}
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          onKeyDown={e => {
            if (e.key === "Enter" || e.key === " ") fileRef.current?.click()
          }}
          className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border p-12 transition-colors hover:border-primary"
        >
          <Upload className="mb-4 h-10 w-10 text-muted-foreground" />
          <p className="text-lg font-medium">Перетащите видео сюда или нажмите для выбора</p>
          <p className="mt-1 text-sm text-muted-foreground">MP4, MOV, WebM</p>
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={e => {
              const f = e.target.files?.[0]
              if (f) handleFile(f)
            }}
          />
        </div>
      )}

      {/* Loading */}
      {isAnalyzing && (
        <div className="flex flex-col items-center justify-center gap-4 py-20">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Обнаружение людей...</p>
        </div>
      )}

      {/* Error */}
      {status === "error" && (
        <div className="flex flex-col items-center gap-4 py-10">
          <AlertCircle className="h-8 w-8 text-destructive" />
          <p className="text-destructive">{error}</p>
          <Button variant="outline" onClick={() => setStatus("idle")}>
            Попробовать снова
          </Button>
        </div>
      )}

      {/* Ready — two-panel layout */}
      {status === "ready" && detectResult && (
        <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
          {/* Left: preview */}
          <Card>
            <CardContent className="p-4">
              <div className="mb-3 flex items-center gap-2">
                {selectedPerson !== null && <CheckCircle className="h-4 w-4 text-green-500" />}
                <span className="text-sm">{detectResult.status}</span>
              </div>
              {/* biome-ignore lint/a11y/useSemanticElements: div needed as positioned container */}
              <div
                role="button"
                tabIndex={0}
                onClick={pickPerson}
                onKeyDown={e => {
                  if (e.key === "Enter" || e.key === " ") {
                    const rect = imgRef.current?.getBoundingClientRect()
                    if (rect) pickPersonAt(rect.left + rect.width / 2, rect.top + rect.height / 2)
                  }
                }}
                className="relative w-full cursor-crosshair"
              >
                <img
                  ref={imgRef}
                  src={previewSrc}
                  alt="Превью"
                  className="w-full rounded border border-border"
                  draggable={false}
                />
                {bboxStyle && (
                  <div
                    className="pointer-events-none absolute border-2 border-green-500 rounded-sm"
                    style={bboxStyle}
                  />
                )}
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                Нажмите на фигуриста на превью для выбора
              </p>
            </CardContent>
          </Card>

          {/* Right: settings */}
          <div className="flex flex-col gap-4">
            {/* Person list */}
            {detectResult.persons.length > 1 && (
              <Card>
                <CardContent className="p-4">
                  <h3 className="mb-2 text-sm font-medium">Фигуристы</h3>
                  <div className="flex flex-col gap-1">
                    {detectResult.persons.map(p => (
                      <button
                        type="button"
                        key={p.track_id}
                        onClick={() => selectFromList(p.track_id)}
                        className={`rounded px-3 py-1.5 text-left text-sm transition-colors ${
                          selectedPerson === p.track_id
                            ? "bg-primary text-primary-foreground"
                            : "hover:bg-muted"
                        }`}
                      >
                        #{p.track_id} — {p.hits} кадров
                      </button>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Settings */}
            <Card>
              <CardContent className="space-y-4 p-4">
                <h3 className="text-sm font-medium">Настройки</h3>

                <div>
                  <span className="mb-1 block text-xs text-muted-foreground">
                    Frame skip: {frameSkip}
                  </span>
                  <Slider
                    value={[frameSkip]}
                    onValueChange={([v]) => setFrameSkip(v)}
                    min={1}
                    max={8}
                    step={1}
                  />
                </div>

                <div>
                  <span className="mb-1 block text-xs text-muted-foreground">
                    HUD Layer: {layer}
                  </span>
                  <Slider
                    value={[layer]}
                    onValueChange={([v]) => setLayer(v)}
                    min={0}
                    max={3}
                    step={1}
                  />
                </div>

                <div>
                  <span className="mb-1 block text-xs text-muted-foreground">Трекинг</span>
                  <Select value={tracking} onValueChange={setTracking}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Авто</SelectItem>
                      <SelectItem value="none">Без трекинга</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center gap-2">
                  <Checkbox
                    id="export"
                    checked={doExport}
                    onCheckedChange={v => setDoExport(v === true)}
                  />
                  <label htmlFor="export" className="text-sm">
                    Экспорт поз + CSV
                  </label>
                </div>
              </CardContent>
            </Card>

            {/* Actions */}
            <div className="flex flex-col gap-2">
              <Button onClick={handleAnalyze} disabled={clickCoord === null} size="lg">
                Анализировать
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setStatus("idle")
                  setDetectResult(null)
                  setClickCoord(null)
                  setSelectedPerson(null)
                }}
              >
                Другое видео
              </Button>
            </div>

            {/* File info */}
            {file && <p className="text-xs text-muted-foreground">{file.name}</p>}
          </div>
        </div>
      )}
    </div>
  )
}
