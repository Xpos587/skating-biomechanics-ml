import { Card, CardContent } from "@/components/ui/card"

interface VideoPlayerProps {
  src: string
}

export function VideoPlayer({ src }: VideoPlayerProps) {
  return (
    <Card>
      <CardContent className="p-4">
        {/* biome-ignore lint/a11y/useMediaCaption: analysis output, not media */}
        <video src={src} controls className="w-full rounded-none" />
      </CardContent>
    </Card>
  )
}
